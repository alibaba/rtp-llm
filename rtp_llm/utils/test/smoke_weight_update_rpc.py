"""Exercise the existing RL client's all-rank update contract on a test server."""

import time

from rtp_llm.RL.rl_inference_client import AsyncRpcInferenceClient


def rank_addresses(port, server_config, world_size):
    # This smoke launches local engines. Use the server's port layout
    # (including --worker_info_port_num), not adjacent ports or only the leader.
    if world_size < 1:
        raise ValueError("CUDA IPC smoke requires at least one engine")
    server_config.start_port = port
    members = []
    for rank in range(world_size):
        server_config.set_local_rank(rank)
        members.append(f"127.0.0.1:{server_config.rpc_server_port}")
    return members


async def paused_weight_update(members, name, desc, timeout):
    if not members or len(set(members)) != len(members):
        raise ValueError("Expected distinct local engine ranks")
    client = AsyncRpcInferenceClient(members)
    deadline = time.monotonic() + timeout

    async def require_all(stage, rpc, *args):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out before {stage}")
        results = await rpc(*args, timeout=remaining)
        failed = [address for address in members if results.get(address) is not True]
        if failed:
            raise RuntimeError(f"{stage} failed on {failed}")

    # On ANY failure the smoke's existing finally stops its own server. Never
    # restart an engine whose peer may still have old/unknown weights.
    await require_all("pause", client.pause_servers)
    await require_all("update", client.update_weights, name, desc, "cuda_ipc")
    await require_all("restart", client.restart_servers)
