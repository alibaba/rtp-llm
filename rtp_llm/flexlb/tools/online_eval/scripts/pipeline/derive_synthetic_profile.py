#!/usr/bin/env python3
"""从已准入 lineage 数据标定 synthetic profile；输出长度独立配置。"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from traffic.datasets import CapturedDataset
from traffic.synthetic.derivation import derive_parameters


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--fit-report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    capture = CapturedDataset.read(args.model)
    raw = args.fit_report.read_bytes()
    result = derive_parameters(capture.model, json.loads(raw), hashlib.sha256(raw).hexdigest(),
                               model_sha256=capture.sha256)
    result['source_capture'] = os.path.relpath(capture.path, args.out.resolve().parent)
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
