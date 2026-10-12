#!/usr/bin/env python3
"""校验已有性能采集包；不连接 Whale，不采样，不修改证据。"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import json
from artifacts.performance_capture import read_manifest, write_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--publish', type=Path, help='将输入声明发布为新包 manifest，证据必须已存在；禁止覆盖')
    args = parser.parse_args(argv)
    if args.publish:
        write_manifest(args.publish, json.loads(args.manifest.read_text()))
    document = read_manifest(args.publish or args.manifest)
    print(document['identity']['id'], document['collection']['status'])


if __name__ == '__main__':
    main()
