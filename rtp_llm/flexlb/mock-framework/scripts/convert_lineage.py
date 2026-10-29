#!/usr/bin/env python3
"""Offline conversion to the sole runtime input codec; lineage is never read by Client."""
import argparse
import hashlib
import json
from itertools import chain, repeat
from pathlib import Path
import sys

FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FRAMEWORK / 'python'), str(FRAMEWORK.parent / 'tools/online_eval/src')]
from mock_framework import client_input_pb2 as pb
from mock_framework.input import write_inputs
from traffic.datasets import CapturedDataset
from traffic.lineage_model import LineageExpander
from traffic.output_sampling import output_sampler


def convert(source, output, *, output_cap, output_mean, seed, priority):
    source, output = Path(source), Path(output)
    manifest_path = output.with_name(output.name.removesuffix('.binpb.zst') + '.manifest.json')
    if output.exists() or manifest_path.exists():
        raise ValueError('Conversion output and manifest must be fresh')
    policy = dict(output_tokens=output_cap, output_distribution=dict(kind='geometric', mean_tokens=output_mean, seed=seed))
    sample = output_sampler(policy)
    captured = CapturedDataset.read(source)
    block = captured.model.metadata.block_size
    origin = captured.model.events[0].timestamp_ms
    count = total_tokens = last = 0

    def requests():
        nonlocal count, total_tokens, last
        for request in LineageExpander(captured.model):
            # Each lineage label encodes a block of repeated synthetic token IDs, as in the original projector.
            tokens = list(chain.from_iterable(repeat(label, block) for label in request.block_labels))
            del tokens[request.input_tokens:]
            last = request.timestamp_ms - origin
            yield pb.Request(token_ids=tokens, max_new_tokens=sample(request.source_index),
                             priority=priority, arrival_offset_ms=last)
            count += 1
            total_tokens += request.input_tokens
            if count % 50000 == 0:
                print(f'converted {count}/{len(captured.model.events)}', flush=True)
    wrote_output = wrote_manifest = False
    try:
        write_inputs(output, requests())
        wrote_output = True
        with output.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest = dict(format='MCKINP01/Protobuf+Zstd', source=source.name, source_sha256=captured.sha256,
            sha256=digest, requests=count, input_tokens=total_tokens, source_span_ms=last,
            arrival='source_relative_ms_before_replay_speed', priority=priority, **policy,
            limitations=captured.manifest['limitations'])
        with manifest_path.open('x') as stream:
            wrote_manifest = True
            json.dump(manifest, stream, ensure_ascii=False, indent=2); stream.write('\n')
        return manifest
    except BaseException:
        if wrote_output:
            output.unlink(missing_ok=True)
        if wrote_manifest:
            manifest_path.unlink(missing_ok=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--output-cap', type=int, required=True)
    parser.add_argument('--output-mean', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--priority', type=int, default=50)
    args = vars(parser.parse_args())
    print(json.dumps(convert(**args), ensure_ascii=False))
