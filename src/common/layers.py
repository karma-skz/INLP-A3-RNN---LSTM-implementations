from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualRNNDecryptor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        output_vocab_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        output_vocab_size = vocab_size if output_vocab_size is None else output_vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.hidden_dropout = nn.Dropout(dropout)
        self.cells = nn.ModuleList()
        in_dim = emb_dim
        for _ in range(num_layers):
            self.cells.append(_ManualRNNCell(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.output_norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps = x.shape
        states = [torch.zeros(bsz, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        embs = self.input_dropout(self.embedding(x))
        outs = []
        for t in range(steps):
            layer_input = embs[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                states[layer_idx] = cell(layer_input, states[layer_idx])
                layer_input = states[layer_idx]
                if layer_idx + 1 < self.num_layers:
                    layer_input = self.hidden_dropout(layer_input)
            outs.append(self.proj(self.output_norm(layer_input)))
        return torch.stack(outs, dim=1)


class ManualLSTMDecryptor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        output_vocab_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        output_vocab_size = vocab_size if output_vocab_size is None else output_vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.hidden_dropout = nn.Dropout(dropout)
        self.cells = nn.ModuleList()
        in_dim = emb_dim
        for _ in range(num_layers):
            self.cells.append(_ManualLSTMCell(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.output_norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps = x.shape
        h_states = [torch.zeros(bsz, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        c_states = [torch.zeros(bsz, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        embs = self.input_dropout(self.embedding(x))
        outs = []
        for t in range(steps):
            layer_input = embs[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h_states[layer_idx], c_states[layer_idx] = cell(
                    layer_input,
                    h_states[layer_idx],
                    c_states[layer_idx],
                )
                layer_input = h_states[layer_idx]
                if layer_idx + 1 < self.num_layers:
                    layer_input = self.hidden_dropout(layer_input)
            outs.append(self.proj(self.output_norm(layer_input)))
        return torch.stack(outs, dim=1)


class SimpleSSM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        state_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.input_proj = nn.Identity() if emb_dim == state_dim else nn.Linear(emb_dim, state_dim)
        self.blocks = nn.ModuleList(
            _DiagonalSSMBlock(model_dim=state_dim, dropout=dropout) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(state_dim)
        self.out = nn.Linear(state_dim, vocab_size, bias=False)
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))
        self.state_dim = state_dim
        self.tie_embeddings = tie_embeddings and emb_dim == state_dim
        if self.tie_embeddings:
            self.out.weight = self.embedding.weight

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        if isinstance(self.input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
        self.final_norm.reset_parameters()
        if not self.tie_embeddings:
            nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(x)
        hidden = self.input_proj(hidden)
        hidden = self.dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden[:, -1, :])
        return self.out(hidden) + self.out_bias


class _DiagonalSSMBlock(nn.Module):
    def __init__(self, model_dim: int, dropout: float):
        super().__init__()
        self.model_dim = model_dim
        self.norm = nn.LayerNorm(model_dim)
        self.b = nn.Parameter(torch.empty(model_dim, model_dim))
        self.c = nn.Parameter(torch.empty(model_dim, model_dim))
        self.d = nn.Linear(model_dim, model_dim)
        self.a_log = nn.Parameter(torch.log(torch.linspace(0.5, 4.0, model_dim)))
        self.dt_log = nn.Parameter(torch.full((model_dim,), -2.0))
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        nn.init.xavier_uniform_(self.b)
        nn.init.xavier_uniform_(self.c)
        nn.init.xavier_uniform_(self.d.weight)
        nn.init.zeros_(self.d.bias)

    def _discretize(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Bilinear (trapezoidal) discretization of a stable diagonal SSM.
        a = -torch.exp(self.a_log)
        dt = F.softplus(self.dt_log) + 1e-4
        half = 0.5 * dt * a
        a_bar = (1.0 + half) / (1.0 - half)
        b_bar = (dt / (1.0 - half)).unsqueeze(-1) * self.b
        return a_bar, b_bar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        u = self.norm(x)
        a_bar, b_bar = self._discretize()
        state = u.new_zeros(u.size(0), self.model_dim)
        outs = []
        for t in range(u.size(1)):
            ut = u[:, t, :]
            state = state * a_bar + ut @ b_bar.T
            yt = state @ self.c.T + self.d(ut)
            outs.append(yt)
        y = torch.stack(outs, dim=1)
        return residual + self.dropout(y)


class _ManualLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_ifog = nn.Linear(input_dim + hidden_dim, hidden_dim * 4)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_ifog.weight)
        nn.init.zeros_(self.w_ifog.bias)
        with torch.no_grad():
            self.w_ifog.bias[self.hidden_dim : self.hidden_dim * 2].fill_(1.0)

    def forward(self, xt: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        z = torch.cat([xt, h], dim=-1)
        i, f, o, g = self.w_ifog(z).chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class _ManualRNNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.orthogonal_(self.hidden_proj.weight)

    def forward(self, xt: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.input_proj(xt) + self.hidden_proj(h))


class ManualBiLSTMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)
        self.layer_dropout = nn.Dropout(dropout)
        self.fwd_layers = nn.ModuleList()
        self.bwd_layers = nn.ModuleList()
        input_dim = emb_dim
        for _ in range(num_layers):
            self.fwd_layers.append(_ManualLSTMCell(input_dim, hidden_dim))
            self.bwd_layers.append(_ManualLSTMCell(input_dim, hidden_dim))
            input_dim = hidden_dim * 2
        self.init_h_f = nn.Parameter(torch.empty(num_layers, hidden_dim))
        self.init_c_f = nn.Parameter(torch.empty(num_layers, hidden_dim))
        self.init_h_b = nn.Parameter(torch.empty(num_layers, hidden_dim))
        self.init_c_b = nn.Parameter(torch.empty(num_layers, hidden_dim))
        self.attn_query = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attn_out = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.output_norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Linear(hidden_dim * 2, emb_dim)
        self.head_norm = nn.LayerNorm(emb_dim)
        self.proj = nn.Linear(emb_dim, vocab_size, bias=False)
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attn_scale = (hidden_dim * 2) ** -0.5
        self.tie_embeddings = tie_embeddings
        if self.tie_embeddings:
            self.proj.weight = self.embedding.weight

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        nn.init.uniform_(self.init_h_f, -1e-2, 1e-2)
        nn.init.uniform_(self.init_c_f, -1e-2, 1e-2)
        nn.init.uniform_(self.init_h_b, -1e-2, 1e-2)
        nn.init.uniform_(self.init_c_b, -1e-2, 1e-2)
        nn.init.xavier_uniform_(self.attn_query.weight)
        nn.init.zeros_(self.attn_query.bias)
        nn.init.xavier_uniform_(self.attn_out.weight)
        nn.init.zeros_(self.attn_out.bias)
        self.output_norm.reset_parameters()
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.head_norm.reset_parameters()
        if not self.tie_embeddings:
            nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.out_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps = x.shape
        lengths = x.ne(0).sum(dim=1)
        valid_mask = x.ne(0).unsqueeze(-1)
        layer_input = self.input_dropout(self.embedding(x))

        for layer_idx in range(self.num_layers):
            h_f = self.init_h_f[layer_idx].unsqueeze(0).expand(bsz, -1)
            c_f = self.init_c_f[layer_idx].unsqueeze(0).expand(bsz, -1)
            f_states = []
            for t in range(steps):
                next_h_f, next_c_f = self.fwd_layers[layer_idx](layer_input[:, t, :], h_f, c_f)
                active = (t < lengths).unsqueeze(-1)
                h_f = torch.where(active, next_h_f, h_f)
                c_f = torch.where(active, next_c_f, c_f)
                f_states.append(h_f)

            h_b = self.init_h_b[layer_idx].unsqueeze(0).expand(bsz, -1)
            c_b = self.init_c_b[layer_idx].unsqueeze(0).expand(bsz, -1)
            b_states = [None] * steps
            for t in range(steps - 1, -1, -1):
                next_h_b, next_c_b = self.bwd_layers[layer_idx](layer_input[:, t, :], h_b, c_b)
                active = (t < lengths).unsqueeze(-1)
                h_b = torch.where(active, next_h_b, h_b)
                c_b = torch.where(active, next_c_b, c_b)
                b_states[t] = h_b

            layer_output = torch.stack(
                [torch.cat([f_states[t], b_states[t]], dim=-1) for t in range(steps)],
                dim=1,
            )
            layer_output = layer_output.masked_fill(~valid_mask, 0.0)
            layer_input = self.layer_dropout(layer_output) if layer_idx + 1 < self.num_layers else layer_output

        encoded = layer_input
        query = self.attn_query(encoded)
        scores = torch.matmul(query, encoded.transpose(1, 2)) * self.attn_scale
        key_mask = x.ne(0).unsqueeze(1)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        scores = scores.masked_fill(~valid_mask, 0.0)
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(~valid_mask, 0.0)
        context = torch.matmul(attn, encoded)
        refined = encoded + self.layer_dropout(torch.tanh(self.attn_out(torch.cat([encoded, context], dim=-1))))
        refined = self.output_norm(refined)
        refined = refined.masked_fill(~valid_mask, 0.0)
        head = self.head_norm(F.gelu(self.head(refined)))
        return self.proj(head) + self.out_bias
