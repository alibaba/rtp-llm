def config(biz_name=""):
    server_ip = "10.0.0.1" if biz_name != "prefill_biz_1" else ""
    server_port = 19000 if biz_name == "prefill_biz_1" else 18000
    return {
        "kernel_list": [
            "navi_sched.agent_report_source_k",
            "navi_sched.agent_pending_cache_sync_k",
            "navi_sched.agent_latency_model_warmup_k",
            "navi_sched.agent_scheduler_config_k",
            "navi_sched.agent_stop_ack_k",
            "navi_sched.agent_request_k",
            "^navi_sched\\.simulate_.*_k$",
        ],
        "resource_config": [
            {
                "name": "navi_sched.agent_cache_r",
                "config": {"total_blocks": 0},
            },
            {
                "name": "navi_sched.agent_report_r",
                "config": {
                    "server_ip": server_ip,
                    "server_port": server_port,
                },
            },
        ],
        "resource_list": [
            "navi_sched.agent_instance_r",
            "navi_sched.agent_scheduler_config_r",
            "navi_sched.agent_report_r",
        ],
    }
