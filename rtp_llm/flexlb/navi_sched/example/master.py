import argparse

from navi_sched import NaviSchedMaster

from common import (
    add_config_arguments,
    role_options,
    runtime_options,
    wait_for_stop,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    parser.add_argument("--master-biz-name", required=True)
    args = parser.parse_args()

    with NaviSchedMaster(
        args.master_biz_name,
        **runtime_options(args, support_heartbeat=True),
        name=args.name,
        **role_options(args),
    ) as master:
        master.init()
        wait_for_stop()


if __name__ == "__main__":
    main()
