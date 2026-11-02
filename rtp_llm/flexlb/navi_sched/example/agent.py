import argparse

from navi_sched import NaviSchedAgent

from common import (
    add_agent_cache_arguments,
    add_config_arguments,
    role_options,
    runtime_options,
    wait_for_stop,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    add_agent_cache_arguments(parser)
    parser.add_argument("--biz-name", required=True)
    args = parser.parse_args()

    with NaviSchedAgent(
        args.biz_name,
        total_blocks=args.total_blocks,
        block_size=args.block_size,
        linear_step=args.linear_step,
        hybrid_tail_required=args.hybrid_tail_required,
        **runtime_options(args),
        name=args.name,
        **role_options(args),
    ) as agent:
        agent.init()
        wait_for_stop()


if __name__ == "__main__":
    main()
