"""Two explicitly experimental EGM tables: raw and preconverted canonical.

No production manifest/backend is accepted or changed. Publication and owner
teardown reuse the tested explicit STOP lifecycle from the integration harness.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE.parent / 'integration'),
               str(HERE.parent / 'native'), str(HERE / 'quant_probe')]
import torch
import owner_fixture
from latency_fixture import raw_rows
from canonical_preprocess import canonicalize_chunks
from build_quant_probe import load_probe, default_build_directory


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--rows', type=int, default=1 << 20)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--variant', choices=('legacy', 'v2'), default='legacy')
    args, remaining = parser.parse_known_args()
    if args.rows < 1 or args.dim not in (32,64,128,256):
        parser.error('positive rows and supported dimension required')
    sys.argv[1:] = remaining
    torch.set_num_threads(4)

    def create(native, owners, device, numa):
        total = args.rows * (args.dim + args.dim // 32)
        if total > native.max_allocation_bytes:
            raise ValueError('experimental table exceeds 1GiB native allocation limit')
        quant = load_probe(str(REPO), default_build_directory())
        tables = {}
        digests = {}
        for name in ('raw', 'canonical'):
            tables[name] = native.Owner(size=total, numa=numa, device=device)
            owners.append(tables[name])
            digests[name] = [hashlib.sha256(), hashlib.sha256()]
        chunk_rows = 16384
        offset_scale = args.rows * args.dim
        for begin in range(0, args.rows, chunk_rows):
            end = min(begin + chunk_rows, args.rows)
            raw_weight, raw_scale = raw_rows(torch.arange(begin, end), args.dim)
            canonical = next(canonicalize_chunks(raw_weight, raw_scale, variant=args.variant,
                             quant_module=quant, device=torch.device('cuda', device),
                             chunk_rows=chunk_rows))
            for name, tensors in [('raw', (raw_weight, raw_scale)),
                                  ('canonical', (canonical.q_bytes, canonical.scale_bytes))]:
                for index, (tensor, offset) in enumerate(zip(tensors,
                        [begin * args.dim, offset_scale + begin * (args.dim // 32)])):
                    payload = tensor.numpy().tobytes()
                    digests[name][index].update(payload)
                    tables[name].load(payload, offset=offset)
        generation = str(uuid4())
        document = dict(experiment_schema=1, experiment='engram-latency-canonical-v0',
                        generation=generation, dataset_id='fixture-r17-c13-mod31-s7',
                        rows=args.rows, dim=args.dim, variant=args.variant,
                        tables={})
        capabilities, descriptions = {}, []
        for name, table in tables.items():
            info = table.info()
            allocation = str(uuid4())
            document['tables'][name] = dict(allocation_id=allocation,
                allocation_bytes=info['allocation_bytes'], granularity=info['granularity'],
                scale_offset=offset_scale,
                weight_sha256=digests[name][0].hexdigest(), scale_sha256=digests[name][1].hexdigest())
            capabilities[name] = base64.b64encode(table.export_handle()).decode('ascii')
            descriptions.append(dict(format=name, allocation_id=allocation,
                                     allocation_bytes=info['allocation_bytes']))
        owner_fixture.emit('EXPERIMENT_TABLES_LOADED', rows=args.rows, dim=args.dim,
                           variant=args.variant, logical_bytes_per_table=total)
        return document, capabilities, descriptions

    return owner_fixture.main(create)


if __name__ == '__main__':
    raise SystemExit(main())
