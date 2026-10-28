import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .config import ConfigError, Registry, compile_case
from .config import load_yaml
from .execution import ExecutionError, prepare_launch, run_case


def main():
    parser = argparse.ArgumentParser(description='Compile or execute a Mock case')
    parser.add_argument('command', choices=['compile', 'run'])
    parser.add_argument('case', type=Path)
    parser.add_argument('--shared-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--launch', type=Path, help='Developer component launch recipes')
    args = parser.parse_args()
    try:
        registry = Registry.standard()
        case = compile_case(args.case, registry, shared_root=args.shared_root)
        if args.command == 'run':
            if args.output is None or args.launch is None:
                parser.error('run requires --output <new directory> and --launch <recipes.yaml>')
            launch = prepare_launch(load_yaml(args.launch), case, args.launch.resolve().parent)
            result = asyncio.run(run_case(case, registry, launch, args.output))
            sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            if result['status'] != 'PASS':
                parser.exit(1)
            return
        if args.launch is not None:
            parser.error('--launch is only valid with run')
        encoded = json.dumps(asdict(case), default=str, ensure_ascii=False, indent=2) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding='utf-8')
        else:
            sys.stdout.write(encoded)
    except (ConfigError, ExecutionError, OSError, ValueError) as error:
        parser.exit(2, f'Configuration error: {error}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Execution cancelled\n')


if __name__ == '__main__':
    main()
