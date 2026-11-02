import argparse

from navi_sched import (
    NaviSchedClient,
    NaviSchedRequest,
)

from common import add_config_arguments, role_options, runtime_options


def parse_hashes(value: str):
    return [int(block_hash) for block_hash in value.split(",") if block_hash]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    parser.add_argument("--master-biz-name", required=True)
    parser.add_argument("--role-biz-name", default="navi_sched_client")
    parser.add_argument("--biz-name", required=True)
    parser.add_argument("--request-id", type=int, default=1)
    parser.add_argument("--prompt-token-count", required=True, type=int)
    parser.add_argument("--block-hashes", default="")
    parser.add_argument("--timeout-ms", type=int, default=1000)
    args = parser.parse_args()

    request = NaviSchedRequest(
        request_id=args.request_id,
        prompt_token_count=args.prompt_token_count,
        block_hashes=parse_hashes(args.block_hashes),
    )
    with NaviSchedClient(
        args.role_biz_name,
        args.master_biz_name,
        name=args.name,
        **runtime_options(args),
        **role_options(args),
    ) as client:
        client.init()
        decision = client.schedule(
            request,
            target_biz_name=args.biz_name,
            timeout_ms=args.timeout_ms,
        )
        print(decision)


if __name__ == "__main__":
    main()
