from __future__ import annotations

from dataclasses import replace

from rtp_llm.kv_cache_subscriber.config import SubscriberConfig


def make_config(**overrides: object) -> SubscriberConfig:
    config = SubscriberConfig(
        rtp_endpoints=("127.0.0.1:8089",),
        rtp_rpc_timeout_s=1.0,
        poll_interval_s=1.0,
        deletion_confirmations=2,
        engine_failure_threshold=2,
        full_refresh_interval_s=300.0,
        kvcm_url="http://kvcm.test:6382",
        kvcm_request_timeout_s=5.0,
        kvcm_heartbeat_interval_s=1.0,
        kvcm_report_batch_size=1000,
        instance_id="instance-a",
        instance_group="group-a",
        host_ip_port="10.0.0.8:8088",
        storage_type="ST_EVENT_REPORT",
        medium="hbm",
        reset_on_start=True,
        log_level="INFO",
    )
    return replace(config, **overrides)
