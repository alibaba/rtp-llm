"""Engram hashing from explicit canonical request history, without slot state."""

import numpy as np
import torch
from sympy import isprime
from torch import nn


def build_compressed_token_map(tokenizer) -> list[int]:
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    mapping, known = [], {}
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            key = normalizer.normalize_str(text) or text
        if key not in known:
            known[key] = len(known)
        mapping.append(known[key])
    return mapping


class EngramHash(nn.Module):
    def __init__(self, config, token_map: list[int]):
        super().__init__()
        t = config.text
        if (
            not token_map
            or len(token_map) != t.get("vocab_size", len(token_map))
            or any(type(value) is not int or value < 0 for value in token_map)
        ):
            raise ValueError("Engram token map must cover the canonical vocabulary")
        vocabulary = max(token_map) + 1
        if (
            vocabulary != t["engram_compressed_vocab_size"]
            or len(set(token_map)) != vocabulary
        ):
            raise ValueError(
                "compressed tokenizer vocabulary does not match Engram weights"
            )
        self.pad_token_id = t["engram_pad_token_id"]
        self.pad_id = token_map[self.pad_token_id]
        self.lookback = t["engram_max_ngram_size"] - 1
        seen, primes, offsets, multipliers = set(), [], [], []
        bound = max(1, (np.iinfo(np.int64).max // vocabulary) // 2)
        for layer, rows in zip(t["engram_layer_ids"], t["engram_num_embeddings"]):
            layer_primes = []
            for _ in range(self.lookback):
                sizes, current = [], t["engram_vocab_size"] - 1
                for _ in range(t["engram_n_heads"]):
                    current += 1
                    while not isprime(current) or current in seen:
                        current += 1
                    seen.add(current)
                    sizes.append(current)
                layer_primes.append(sizes)
            flat = [prime for sizes in layer_primes for prime in sizes]
            if sum(flat) != rows:
                raise ValueError(
                    f"Engram table L{layer} row count does not match its prime ranges"
                )
            primes.append(layer_primes)
            offsets.append(np.cumsum([0, *flat[:-1]]))
            rng = np.random.default_rng(10007 * layer)
            multipliers.append(
                rng.integers(0, bound, size=self.lookback + 1, dtype=np.int64) * 2 + 1
            )
        self.register_buffer(
            "token_map", torch.tensor(token_map, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "primes", torch.tensor(primes, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "offsets", torch.tensor(np.array(offsets)), persistent=False
        )
        self.register_buffer(
            "multipliers", torch.tensor(np.array(multipliers)), persistent=False
        )

    def forward(self, input_ids, history_ids, history_valid, token_mask=None):
        """Hash contiguous [B,L] input before CP remapping.

        history_ids/history_valid are [B,3] canonical predecessors; invalid
        entries include sequence padding and image spans. This call never
        commits history, so rejected draft tokens cannot contaminate a slot.
        """
        if input_ids.dtype != torch.int64 or history_ids.dtype != torch.int64:
            raise ValueError("Engram requires canonical int64 token IDs")
        batch, length = input_ids.shape
        if (
            history_ids.shape != (batch, self.lookback)
            or history_valid.shape != history_ids.shape
            or history_valid.dtype != torch.bool
        ):
            raise ValueError(
                "Engram history must contain exactly three canonical predecessors"
            )
        if token_mask is None:
            token_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if token_mask.shape != input_ids.shape or token_mask.dtype != torch.bool:
            raise ValueError("Engram token mask must include the complete image spans")
        all_ids = torch.cat((history_ids, input_ids), dim=1)
        all_valid = torch.cat((history_valid, token_mask), dim=1)
        safe_ids = all_ids.masked_fill(~all_valid, self.pad_token_id)
        torch._assert_async(
            (safe_ids >= 0).all(), "Engram received a negative canonical token ID"
        )
        torch._assert_async(
            (safe_ids < self.token_map.numel()).all(),
            "Engram received an out-of-vocabulary canonical token ID",
        )
        compressed = self.token_map[safe_ids]
        blocked = torch.zeros_like(input_ids, dtype=torch.bool)
        tokens = []
        for shift in range(self.lookback + 1):
            begin = self.lookback - shift
            blocked = blocked | ~all_valid[:, begin : begin + length]
            tokens.append(
                torch.where(blocked, self.pad_id, compressed[:, begin : begin + length])
            )
        products = torch.stack(tokens, -1).unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for index in range(1, self.lookback + 1):
            rolling = torch.bitwise_xor(rolling, products[..., index])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, index - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


def committed_history(history_ids, history_valid, accepted_ids, accepted_valid):
    """Return the next boundary from accepted canonical rows only."""
    if (
        history_ids.shape != history_valid.shape
        or accepted_ids.shape != accepted_valid.shape
    ):
        raise ValueError("history IDs and validity shapes must match")
    if (
        history_ids.ndim != 2
        or history_ids.shape[1] != 3
        or accepted_ids.ndim != 2
        or accepted_ids.shape[0] != history_ids.shape[0]
    ):
        raise ValueError("history is [B,3] and accepted tokens are [B,L]")
    return (
        torch.cat((history_ids, accepted_ids), 1)[:, -3:].clone(),
        torch.cat((history_valid, accepted_valid), 1)[:, -3:].clone(),
    )


class Engram(nn.Module):
    """Local lookup, model-supplied group32 projection, and per-HC gate injection."""

    @classmethod
    def from_weights(cls, layer_id, weights, shared_lookup, *, projection_factory=None):
        """Bind checkpoint-local projection/gates while the caller owns host tables."""
        from rtp_llm.models_py.modules.dsv41.linear import V41Block32Linear

        if any(name.startswith("engram.embed.") for name in weights):
            raise ValueError("Engram tables must remain in the host-shared loader")
        q_weight, k_weight = weights["engram.q_weight"], weights["engram.k_weight"]
        if (
            q_weight.ndim != 2
            or q_weight.shape[0] != 4
            or q_weight.shape != k_weight.shape
            or q_weight.dtype != torch.bfloat16
            or k_weight.dtype != torch.bfloat16
            or q_weight.device != k_weight.device
        ):
            raise ValueError("Engram requires checkpoint BF16 [4,hidden] gate weights")
        weight, scale = weights["engram.wkv.weight"], weights["engram.wkv.scale"]
        output_dim = 5 * q_weight.shape[1]
        if (
            weight.shape != (output_dim, 24 * 256)
            or weight.dtype != torch.float8_e4m3fn
            or weight.device != q_weight.device
            or scale.shape != (output_dim // 32, 192)
            or scale.dtype != torch.float8_e8m0fnu
            or scale.device != weight.device
        ):
            raise ValueError(
                "Engram projection must retain its checkpoint block32 layout"
            )
        factory = V41Block32Linear if projection_factory is None else projection_factory
        return cls(layer_id, shared_lookup, factory(weight, scale), q_weight, k_weight)

    def __init__(self, layer_id, shared_lookup, projection, q_weight, k_weight):
        super().__init__()
        if (
            layer_id not in (1, 14)
            or q_weight.shape != k_weight.shape
            or q_weight.ndim != 2
        ):
            raise ValueError(
                "Engram requires an owner layer and matching [HC, hidden] gate weights"
            )
        self.layer_id = layer_id
        # Deliberately not a Tensor/buffer: Module.to() must never migrate the tables.
        self.shared_lookup = shared_lookup
        self.projection = projection
        self.q_weight = nn.Parameter(q_weight, requires_grad=False)
        self.k_weight = nn.Parameter(k_weight, requires_grad=False)

    def forward(self, hidden, hash_ids, token_mask=None, *, lookup_output=None):
        from rtp_llm.models_py.modules.dsv41.math import engram_inject

        if (
            hash_ids.shape != hidden.shape[:-2] + (24,)
            or hidden.shape[-2:] != self.q_weight.shape
        ):
            raise ValueError(
                "Engram hashes must contain 24 heads for each canonical token"
            )
        valid = None
        if token_mask is not None:
            if token_mask.shape != hidden.shape[:-2] or token_mask.dtype != torch.bool:
                raise ValueError("Engram token mask must cover complete image spans")
            valid = token_mask.unsqueeze(-1).expand_as(hash_ids).contiguous()
        rows = self.shared_lookup.lookup(
            self.layer_id, hash_ids, valid_mask=valid, out=lookup_output
        )
        projected = self.projection(rows.flatten(-2))
        return engram_inject(
            hidden, projected, self.q_weight, self.k_weight, token_mask
        )
