"""Compare saved, matching-input native / inference logits (diagnostics only)."""
import argparse
import gzip
import json
import math
from pathlib import Path


def metrics(a, b):
    if not a or len(a) != len(b):
        raise ValueError("nonempty, equal vocabulary dimensions required")
    if not all(math.isfinite(v) for v in a + b):
        raise ValueError("nonfinite logits")
    def log_probs(values):
        peak = max(values)
        norm = peak + math.log(sum(math.exp(v - peak) for v in values))
        return [v - norm for v in values]
    lp, lq = log_probs(a), log_probs(b)
    norm_a = sum(x*x for x in a)
    norm_b = sum(x*x for x in b)
    return dict(vocabulary=len(a), top1_equal=max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__),
                max_abs=max(abs(x-y) for x,y in zip(a,b)),
                rmse=math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a)),
                relative_l2=math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/max(norm_a,1e-30)),
                cosine=sum(x*y for x,y in zip(a,b))/max(math.sqrt(norm_a*norm_b),1e-30),
                kl_baseline_candidate=sum(math.exp(x)*(x-y) for x,y in zip(lp,lq)))


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        return json.load(f)


def compare(a, b):
    if not a.get('input_ids') or a['input_ids'] != b.get('input_ids'):
        raise ValueError("input token IDs do not match")
    # A native response reports the final step's logits. Multi-step comparison
    # is valid only if every preceding generated token also matches.
    if not a.get('output_ids') or not b.get('output_ids'):
        raise ValueError("output token IDs are required to establish the prefix")
    if len(a['output_ids']) != len(b['output_ids']):
        raise ValueError("output sequence counts do not match")
    for x,y in zip(a['output_ids'],b['output_ids']):
        if not x or not y or x[:-1] != y[:-1]:
            raise ValueError("generated prefixes diverged; logits are not comparable")
    if len(a.get('logits', [])) != len(a['output_ids']) or len(b.get('logits', [])) != len(b['output_ids']):
        raise ValueError("missing final-step logits")
    return [metrics(x,y) for x,y in zip(a['logits'],b['logits'])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(read(args.baseline),read(args.candidate)),indent=2))


if __name__ == '__main__':
    main()
