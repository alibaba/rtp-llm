"""Command-line entry point for completed experiment playback."""
import argparse
import asyncio
from pathlib import Path
import signal

from .service import serve


async def run(args):
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await serve(args.result, args.output, args.prometheus, args.grafana_home, args.port)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser(description='View a completed Case: configuration, saved gates and Grafana metrics')
    parser.add_argument('result', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New isolated viewer session directory')
    parser.add_argument('--prometheus', type=Path, required=True)
    parser.add_argument('--grafana-home', type=Path, required=True)
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('--port must be between 0 and 65535')
    try:
        asyncio.run(run(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'View error: {error}\n')


if __name__ == '__main__':
    main()
