"""Check that KVCM receives cache events from inference."""

import time


def check_model_events(server_manager, kvcm_server):
    import grpc
    from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub

    instance_id = kvcm_server.client_env()["KV_CACHE_EVENT_INSTANCE_ID"]
    deadline = time.monotonic() + 15
    with grpc.insecure_channel(f"127.0.0.1:{server_manager.port + 1}") as channel:
        stub = RpcServiceStub(channel)
        # All keys come from the model's cache, without synthetic ReportEvent calls.
        keys = []
        while time.monotonic() < deadline and not keys:
            status = stub.GetCacheStatus(pb.CacheVersionPB(latest_cache_version=-1, need_cache_keys=True), timeout=5)
            keys = [key for key, present in status.cache_keys.items() if present]
            if not keys:
                time.sleep(0.05)
        if not keys:
            raise AssertionError("The model did not retain any cache keys for event verification")
        # A single retained key is sufficient to prove the model->Publisher->Manager chain.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            response = kvcm_server.post_json("getHostCacheState", {
                "instance_id": instance_id, "query_type": 2,
                "block_cache_keys": [str(keys[0])], "medium": ["hbm"], "p2p_host_count": 0,
            }, check_status=False)
            code = response.get("header", {}).get("status", {}).get("code")
            if code in (1, "1", "OK"):
                if any(int(host.get("local", 0)) == 1 for host in response.get("hosts", [])):
                    return
            elif code not in (8, "8", "INSTANCE_NOT_EXIST"):
                raise AssertionError(f"Model event lookup failed: {response}")
            time.sleep(0.05)
    raise AssertionError("The model's retained cache key did not appear in KVCM HBM host state")
