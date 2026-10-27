import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .config import ConfigError, Registry, compile_case


def main():
    parser = argparse.ArgumentParser(description='Mock case configuration compiler; does not execute cases')
    parser.add_argument('command', choices=['compile'])
    parser.add_argument('case', type=Path)
    parser.add_argument('--catalog', type=Path, required=True, help='Java-generated interface catalog')
    parser.add_argument('--shared-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        case = compile_case(args.case, Registry.load(args.catalog), shared_root=args.shared_root)
        encoded = json.dumps(asdict(case), default=str, ensure_ascii=False, indent=2) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding='utf-8')
        else:
            sys.stdout.write(encoded)
    except ConfigError as error:
        parser.exit(2, f'Configuration error: {error}\n')


if __name__ == '__main__':
    main()
