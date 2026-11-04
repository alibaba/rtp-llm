import argparse
import signal
import threading

def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", required=True)
    parser.add_argument("--grpc-port", type=int, default=0)
    parser.add_argument("--local-node", action="append", default=[])
    parser.add_argument("--vip-domain", default="")
    parser.add_argument("--vip-jmenv-domain", default="")
    parser.add_argument("--thread-num", type=int, default=64)
    parser.add_argument("--disable-kmon", action="store_true")


def add_agent_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--total-blocks", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--linear-step", type=int, required=True)
    parser.add_argument("--hybrid-tail-required", action="store_true")


def role_options(args: argparse.Namespace):
    return {
        "enable_kmon": not args.disable_kmon,
        "gig": {"grpc_port": args.grpc_port},
    }


def runtime_options(
        args: argparse.Namespace, support_heartbeat: bool = False):
    local_nodes = []
    for value in args.local_node:
        biz_name, ip, port = value.rsplit(":", 2)
        local_nodes.append({
            "biz_name": biz_name,
            "ip": ip,
            "grpc_port": int(port),
            "support_heartbeat": support_heartbeat,
        })
    runtime_config = {
        "thread_num": args.thread_num,
        "local_nodes": local_nodes,
    }
    if args.vip_domain:
        runtime_config["vip_subscription"] = {
            "jmenv_domain": args.vip_jmenv_domain,
            "services": [args.vip_domain],
        }
    return {"runtime_config": runtime_config}


def wait_for_stop() -> None:
    stopped = threading.Event()

    def stop_role(signum, frame):
        stopped.set()

    signal.signal(signal.SIGINT, stop_role)
    signal.signal(signal.SIGTERM, stop_role)
    stopped.wait()
