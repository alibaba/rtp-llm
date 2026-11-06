import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
import yaml

from .config import ConfigError, compile_case, load_yaml
from .execution.step.registry import Registry
from .execution import ExecutionError
from .execution.launch import prepare_launch
from .execution.runner import run_case
from .execution.launch import local_launch


def _launch(args, case, parser):
    if args.launch is not None or args.java is None or args.prometheus is None:
        parser.error('launch requires --java <Java21 binary> and --prometheus <binary>; --launch is not accepted')
    launch = local_launch(case, args.java, args.prometheus, framework=Path(__file__).resolve().parents[2])
    encoded = yaml.safe_dump(launch, allow_unicode=True, sort_keys=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(encoded)
    else:
        sys.stdout.write(encoded)


def _run(args, case, registry, parser):
    if args.output is None or args.launch is None:
        parser.error('run requires --output <new directory> and --launch <recipes.yaml>')
    launch = prepare_launch(load_yaml(args.launch), case, args.launch.resolve().parent)

    async def run():
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
        try:
            return await run_case(case, registry, launch, args.output)
        finally:
            loop.remove_signal_handler(signal.SIGTERM)

    try:
        result = asyncio.run(run())
    except asyncio.CancelledError:
        parser.exit(130, 'Execution cancelled\n')
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    if result['status'] != 'PASS':
        parser.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Compile or execute a Mock case')
    parser.add_argument('command', choices=['compile', 'launch', 'run'])
    parser.add_argument('case', type=Path)
    parser.add_argument('--shared-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--launch', type=Path, help='Developer component launch recipes')
    parser.add_argument('--java', type=Path, help='Java 21 executable for launch generation')
    parser.add_argument('--prometheus', type=Path, help='Prometheus executable for launch generation')
    args = parser.parse_args()
    if args.command != 'launch' and any(value is not None for value in (args.java, args.prometheus)):
        parser.error('--java and --prometheus are only valid with launch')
    try:
        registry = Registry.standard()
        case = compile_case(args.case, registry, shared_root=args.shared_root)
        if args.command == 'launch':
            _launch(args, case, parser)
            return
        if args.command == 'run':
            _run(args, case, registry, parser)
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
