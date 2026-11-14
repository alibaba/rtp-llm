import torch

_REAL_VOCAB_SIZES = [
    20000003,
    20000023,
    20000033,
    20000047,
    20000059,
    20000063,
    20000069,
    20000077,
    20000081,
    20000093,
    20000107,
    20000147,
    20000153,
    20000159,
    20000161,
    20000171,
]
_REAL_OFFSETS = [
    0,
    20000003,
    40000026,
    60000059,
    80000106,
    100000165,
    120000228,
    140000297,
    160000374,
    180000455,
    200000548,
    220000655,
    240000802,
    260000955,
    280001114,
    300001275,
]
_REAL_MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]


def _ref_shift(token_ids, shift, eos):
    if shift == 0:
        return token_ids
    batch_size, seq_len = token_ids.shape
    positions = torch.arange(seq_len, dtype=torch.long)
    eos_positions = torch.where(token_ids == eos, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]],
        dim=1,
    )
    position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
    shifted = token_ids.gather(dim=1, index=gather_positions)
    valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, token_ids.new_full((), eos))


def _ref_hashed_ids(
    history, seq_len, vocab_sizes, offsets, multipliers, ngram_size, eos
):
    heads_per_ngram = vocab_sizes.shape[0] // (ngram_size - 1)
    shifted = [_ref_shift(history, s, eos) for s in range(ngram_size)]
    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * multipliers[position])
        head_vocab = vocab_sizes[start:end].view(1, 1, -1)
        ids = torch.remainder(mixed.unsqueeze(-1), head_vocab)
        blocks.append(ids + offsets[start:end].view(1, 1, -1))
    return torch.cat(blocks, dim=-1)[:, -seq_len:]
