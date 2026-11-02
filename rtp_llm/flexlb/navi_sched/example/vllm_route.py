import argparse
import asyncio

from navi_sched.adapters.vllm import NaviSchedAdapterConfig, NaviSchedClient


def parse_token_ids(value):
    return [int(token_id) for token_id in value.split(",") if token_id]


async def route(args):
    config = NaviSchedAdapterConfig.from_env()
    with NaviSchedClient(config) as client:
        return await client.route_async(
            parse_token_ids(args.token_ids),
            request_id=args.request_id,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-ids", required=True)
    parser.add_argument("--request-id", required=True)
    args = parser.parse_args()
    print(asyncio.run(route(args)))


if __name__ == "__main__":
    main()
