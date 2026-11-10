"""Request-major row metadata for inherited target verification."""

from dataclasses import dataclass

@dataclass(frozen=True)
class VerifyLayout:
    lengths: tuple
    prefixes: tuple
    starts: tuple
    rows: int

    @classmethod
    def from_inputs(cls, primary, rows):
        lengths = tuple(int(x) for x in primary.input_lengths.cpu().tolist())
        prefixes = tuple(int(x) for x in primary.prefix_lengths.cpu().tolist())
        return cls.from_lengths_and_prefixes(lengths, prefixes, rows)

    @classmethod
    def from_lengths_and_prefixes(cls, lengths, prefixes, rows):
        if len(prefixes) != len(lengths) or sum(lengths) != rows:
            raise ValueError("verify rows must be request-major dense/packed queries")
        if any(length < 1 for length in lengths) or any(
            prefix < 0 for prefix in prefixes
        ):
            raise ValueError(
                "verify needs an anchor per request and nonnegative prefixes"
            )
        starts, cursor = [], 0
        for length in lengths:
            starts.append(cursor)
            cursor += length
        return cls(lengths, prefixes, tuple(starts), rows)
