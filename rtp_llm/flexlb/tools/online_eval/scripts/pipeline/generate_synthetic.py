#!/usr/bin/env python3
"""离线生成标准合成请求计划，供诊断或直接 Java 播放使用。"""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from traffic.synthetic.generator import materialize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--max-requests', type=int)
    args = parser.parse_args(argv)
    materialize(args.out, json.loads(args.spec.read_text()), args.namespace,
                args.spec.resolve().parent, max_requests=args.max_requests)
    print(args.out.with_suffix('.manifest.json'))


if __name__ == '__main__':
    main()
