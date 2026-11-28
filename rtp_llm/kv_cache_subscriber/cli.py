from __future__ import annotations

import asyncio
import logging

from rtp_llm.kv_cache_subscriber.config import build_parser, config_from_args
from rtp_llm.kv_cache_subscriber.reporter import HttpKvcmReporter
from rtp_llm.kv_cache_subscriber.service import SubscriberService
from rtp_llm.kv_cache_subscriber.source import RtpGrpcCacheStatusSource


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    source = RtpGrpcCacheStatusSource(
        config.rtp_endpoints,
        config.rtp_rpc_timeout_s,
    )
    reporter = HttpKvcmReporter(config)
    asyncio.run(SubscriberService(config, source, reporter).run())
