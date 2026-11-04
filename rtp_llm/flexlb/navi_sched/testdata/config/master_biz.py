def config(biz_name=""):
    return {
        "kernel_list": [
            "navi_sched.master_*",
            "navi_sched.scheduler_config_sync_k",
            "navi_sched.schedule_k",
            "navi_sched.*_scheduler_k",
            "navi_sched.cost_scheduler_*",
            "navi_sched.cost_benchmark_*",
            "navi_sched.fork_test_schedule_k",
        ],
        "resource_config": [],
        "resource_list": [
            "navi_sched.scheduler_config_r",
        ],
    }
