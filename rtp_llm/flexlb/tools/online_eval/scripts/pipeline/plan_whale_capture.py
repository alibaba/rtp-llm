#!/usr/bin/env python3
"""从 Whale MCP 的两份读取结果生成固定版本 GET 计划；不执行网络请求。"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from artifacts.performance_capture import whale_read_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True, help='getPyBizDeploymentById 的 data 对象')
    parser.add_argument('--images', type=Path, required=True, help='deployment_images 返回对象')
    args = parser.parse_args(argv)
    print(json.dumps(whale_read_plan(json.loads(args.deployment.read_text()),
        json.loads(args.images.read_text())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
