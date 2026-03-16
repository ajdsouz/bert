import torch
from torch import nn

from src.modeling._annotation import T, FP, Int, typechecked, tensor_typechecked


@typechecked
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_sequence_length: int, hidden_size: int):
        super().__init__()
        assert hidden_size % 2 == 0
        self._max_sequence_length = max_sequence_length
        self._hidden_size = hidden_size
        # to get devcie and dtype
        self.register_buffer("_dummy_indicator", torch.empty(0), persistent=False)
        self.__pe = None

    @property
    def pe(self):
        # Do not register as buffer in __init__
        # It conflicts with huggingface
        device, dtype = self._dummy_indicator.device, self._dummy_indicator.dtype
        if self.__pe is not None:
            if self.__pe.device != device or self.__pe.dtype != dtype:
                self.__pe = self.__pe.to(device=device, dtype=dtype)
            return self.__pe

        position: FP[T, "s"] = torch.arange(self._max_sequence_length)
        div_terms: FP[T, f"h_2={self._hidden_size//2}"] = torch.exp(
            torch.arange(0, self._hidden_size, 2)
            * (-torch.log(torch.tensor(10000.0)) / self._hidden_size)
        )
        x: FP[T, "s h_2"] = position[:, None] * div_terms[None, :]
        pe: FP[T, "s h"] = torch.zeros(self._max_sequence_length, self._hidden_size)
        pe[:, 0::2] = torch.sin(x)
        pe[:, 1::2] = torch.cos(x)
        self.__pe: FP[T, "s h"] = pe.to(self._dummy_indicator)
        return self.__pe

    @tensor_typechecked
    def forward(
        self, x: FP[T, "batch seq hidden_size"]
    ) -> FP[T, "batch seq hidden_size"]:
        return x + self.pe[None, : x.shape[1], :]


@typechecked
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, gated_attention: bool):
        super().__init__()
        assert hidden_size % num_heads == 0
        self._num_heads = num_heads
        self._size_per_head = hidden_size // num_heads
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
        self.gate_proj = None
        if gated_attention:
            self.gate_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    @tensor_typechecked
    def forward(
        self, x: FP[T, "batch seq hidden_size"], mask: Int[T, "batch seq"]
    ) -> tuple[
        FP[T, "batch seq hidden_size"],
        FP[T, "batch num_heads seq seq"],
    ]:
        """
        Returns:
            a tuple contains
            - `FP[T, "batch seq hidden_size"]` attention value
            - `FP[T, "batch num_heads seq seq"]` attention weight
        """

        B, S, N, D = x.shape[0], x.shape[1], self._num_heads, self._size_per_head
        assert N * D == x.shape[-1]

        def calc(w):
            y: FP[T, f"batch={B} N={N} seq={S} D={D}"] = torch.einsum(
                "bsnd -> bnsd", w(x).reshape(B, S, N, D)
            )
            return y

        q, k, v = (calc(w) for w in (self.wq, self.wk, self.wv))

        scores: FP[T, "batch N seq=seq tar=seq"] = torch.einsum(
            "bnsd,bntd -> bnst", q / (self._size_per_head**0.5), k
        )
        scores = scores.masked_fill(mask.reshape(B, 1, 1, S) == 0, -torch.inf)
        weight: FP[T, "batch N seq=seq tar=seq"] = torch.softmax(scores, -1)

        attention: FP[T, "batch N seq D"] = torch.einsum("bnst,bntd -> bnsd", weight, v)
        attention: FP[T, "batch seq hidden_size"] = torch.einsum(
            "bnsd -> bsnd", attention
        ).reshape(B, S, N * D)

        if self.gate_proj is not None:
            # https://openreview.net/forum?id=1b7whO4SfY
            attention = attention * torch.sigmoid(self.gate_proj(x))

        out = self.out_proj(attention)
        return out, weight


@typechecked
class GatedMlp(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, activation: nn.Module):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.activation = activation

    @tensor_typechecked
    def forward(
        self, x: FP[T, "batch ... hidden_size"]
    ) -> FP[T, "batch ... hidden_size"]:
        x, swish = self.up_proj(x), self.activation(self.gate_proj(x))
        y = self.down_proj(x * swish)
        return y


@typechecked
def make_mlp(mlp_type: str, hidden_size: int) -> nn.Module:
    if mlp_type == "ffn":
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
    elif mlp_type == "gated(gelu)":
        return GatedMlp(hidden_size, hidden_size * 4, nn.GELU())
    else:
        raise NotImplementedError(mlp_type)


@typechecked
class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        gated_attention: bool,
        mlp_type: str,
    ):
        super().__init__()
        self.layernorm_attention = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(
            hidden_size, num_attention_heads, gated_attention
        )
        self.layernorm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = make_mlp(mlp_type, hidden_size)

    @tensor_typechecked
    def forward(
        self, x: FP[T, "batch seq hidden_size"], mask: Int[T, "batch seq"]
    ) -> tuple[
        FP[T, "batch seq hidden_size"],
        FP[T, "batch num_heads seq seq"],
    ]:
        """
        Returns:
            a tuple contains
            - `FP[T, "batch seq hidden_size"]` features of mlp
            - `FP[T, "batch num_heads seq seq"]` attention weight
        """
        attention_value, attention_weight = self.attention(
            self.layernorm_attention(x), mask
        )
        x = x + attention_value
        mlp_out = self.mlp(self.layernorm_mlp(x))
        x = x + mlp_out
        return x, attention_weight
