import json
import os

import file_loader


def get_biz_config(meta_info, config_path):
    biz_config = {}
    for biz_name, value in meta_info["biz_list"].items():
        config = file_loader.load_file(value["biz_config"]).config(biz_name)
        config["part_count"] = value["part_count"]
        config["part_ids"] = value["part_ids"]
        biz_config[biz_name] = config
    return biz_config


def config(config_path, meta_info):
    default_log_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "tmp", "test_cluster"))
    log_dir = os.environ.get("TEST_TMPDIR", default_log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return {
        "version": 0,
        "log_config": {
            "file_appenders": [{
                "file_name": os.path.join(log_dir, meta_info["server_name"]),
                "log_level": "INFO",
                "async_flush": False,
            }],
        },
        "engine_config": {
            "builtin_task_queue": {
                "thread_num": int(os.environ.get(
                    "NAVI_SCHED_TEST_THREAD_NUM", "4")),
                "queue_size": 1000,
                "processing_size": 100_000_000,
            },
            "disable_perf": True,
            "disable_symbol_table": True,
        },
        "config_path": config_path,
        "biz_config": get_biz_config(meta_info, config_path),
        "sleep_before_update_us": 0,
        "init_timeout_ms": 100000,
        "resource_config": [{
            "name": "navi.buildin.gig_client",
            "config": {
                "init_config": {
                    "subscribe_config": {
                        "local": json.loads(meta_info["gig_sub_config"]),
                    },
                    "misc_config": {
                        "snapshot_log_prefix": meta_info["gig_host"],
                        "snapshot_log_dir": log_dir,
                    },
                },
            },
        }],
    }
