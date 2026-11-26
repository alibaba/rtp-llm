"""Generate the startup hook config for a Frontend process profile."""

import argparse
from pathlib import Path

from partitioned_index import durable_json


def configuration(interface, directory, runtime="current"):
    server = "rtp_llm.frontend.frontend_server.FrontendServer"
    visitor = "rtp_llm.server.backend_rpc_server_visitor"
    entries = []

    def add(target, event, callback, **config):
        entries.append({"target": target, event: callback, "config": config})

    if interface in ("http", "both"):
        add(server + ".start", "before", "on_worker_start", protocol="http")
        for method in ("chat_completion", "batch_chat_completion"):
            add(
                server + "." + method, "before", "mark_arrival", interface="openai_http"
            )
        for method in ("inference", "batch_infer"):
            add(
                server + "." + method, "before", "mark_arrival", interface="native_http"
            )
        add(
            "rtp_llm.frontend.frontend_worker.FrontendWorker.__init__",
            "after",
            "seed_frontend_worker",
        )
    if interface in ("grpc", "both"):
        add(
            "rtp_llm.dash_sc.app.DashScApp.start",
            "before",
            "on_worker_start",
            protocol="grpc",
        )
        add(
            "rtp_llm.dash_sc.inference.servicer.emit_query_log",
            "before",
            "mark_arrival",
            interface="dashscope_grpc",
        )
        add(
            "rtp_llm.dash_sc.inference.servicer.extract_span_external_request_id",
            "before",
            "record_source",
        )
    route = (
        "get_master_route_addrs" if runtime == "legacy" else "_get_master_route_addrs"
    )
    for method, callback in (
        ("enqueue", "after_enqueue"),
        ("batch_enqueue", "after_batch"),
        (route, "after_route"),
    ):
        add(visitor + ".BackendRPCServerVisitor." + method, "after", callback)
    add(visitor + ".multimodal_routing_tokens", "after", "after_expand")
    add(visitor + ".get_block_cache_keys", "after", "after_block_keys")
    add(
        "rtp_llm.server.master_client.MasterClient.get_backend_role_addrs",
        "before",
        "before_pd",
    )
    return {
        "hook_file": str(Path(__file__).with_name("hook.py").resolve()),
        "capture": {
            "directory": str(directory),
            "profile": interface,
            "runtime": runtime,
        },
        "function_hooks": entries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interface", choices=("http", "grpc", "both"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--runtime", choices=("current", "legacy"), default="current")
    parser.add_argument(
        "--directory", type=Path, default=Path("/tmp/rtp-frontend-capture")
    )
    args = parser.parse_args()
    durable_json(
        args.output, configuration(args.interface, args.directory, args.runtime)
    )


if __name__ == "__main__":
    main()
