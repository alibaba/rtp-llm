"""Select original DSV4 trace tokens; only manifests are stored in the repo."""

import argparse
import hashlib
import json
from pathlib import Path


def records(path):
    decoder = json.JSONDecoder()
    with path.open() as handle:
        if next(handle).strip() != "[":
            raise ValueError(f"unexpected trace array format: {path}")
        for line in handle:
            if line.strip() == "]":
                break
            row, end = decoder.raw_decode(line)
            if line[end:].strip() not in ("", ","):
                raise ValueError(f"unexpected trailing trace data: {path}")
            yield row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="/home/admin/dataset/dsv4-pro-trace-outlen-100k"
    )
    parser.add_argument("--length", type=int, default=131072)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--output", default="/tmp/dsv4-long-corpus.json")
    args = parser.parse_args()
    selected, seen = [], set()
    scanned = 0
    maximum = 0
    for path in sorted(Path(args.dataset).glob("*/*.json")):
        for index, row in enumerate(records(path)):
            scanned += 1
            ids = row["input_ids"]
            maximum = max(maximum, len(ids))
            if row.get("status") != "OK" or len(ids) < args.length:
                continue
            ids = ids[: args.length]
            digest = hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            selected.append(
                {
                    "input_ids": ids,
                    "source_file": str(path),
                    "source_row": index,
                    "original_input_tokens": row["input_token_len"],
                    "token_sha256": digest,
                }
            )
            if len(selected) == args.count:
                break
        print(
            json.dumps(
                {
                    "file": str(path),
                    "scanned": scanned,
                    "selected": len(selected),
                    "maximum_input_tokens": maximum,
                }
            ),
            flush=True,
        )
        if len(selected) == args.count:
            break
    result = {
        "dataset": args.dataset,
        "length": args.length,
        "scanned": scanned,
        "maximum_input_tokens": maximum,
        "samples": selected,
    }
    Path(args.output).write_text(json.dumps(result))
    if len(selected) != args.count:
        raise RuntimeError(
            f"only {len(selected)} distinct contexts >= {args.length}; maximum {maximum}"
        )


if __name__ == "__main__":
    main()
