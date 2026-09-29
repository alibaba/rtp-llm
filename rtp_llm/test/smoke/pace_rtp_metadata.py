"""Exercise RTP's metadata RPC adapter against its own remote-cache instance."""

import time


def check_metadata_rpc(server_manager, backend_type):
    import grpc
    from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub

    # Negative keys are unique to this run; removal cannot affect model caches.
    key = -time.time_ns()
    with grpc.insecure_channel(f"127.0.0.1:{server_manager.port + 1}") as channel:
        stub = RpcServiceStub(channel)

        def call(operation, query_type=2):
            request = pb.FunctionRequestPB()
            request.remote_request.op = getattr(pb, operation)
            request.remote_request.trace_id = "pace-rtp-metadata-smoke"
            metadata = request.remote_request.metadata
            metadata.query_type = query_type
            metadata.block_keys.append(key)
            metadata.backend_type = backend_type
            metadata.medium.append("hbm")
            response = stub.ExecuteFunction(request, timeout=15)
            if not response.HasField("remote_response"):
                raise AssertionError(f"Missing RTP metadata response for {operation}")
            return response.remote_response

        if call("REMOTE_OPERATION_MATCH_LOCATION_LEN").matched_blocks != 0:
            raise AssertionError("Fresh key unexpectedly matched")
        call("REMOTE_OPERATION_MATCH_LOCATION")
        call("REMOTE_OPERATION_MATCH_META")
        backend = call("REMOTE_OPERATION_GET_LOCATIONS_BY_BACKEND", query_type=1)
        if len(backend.backend_locations) != 1 or backend.backend_locations[0].locations:
            raise AssertionError("RTP backend query did not preserve a miss position")
        if call("REMOTE_OPERATION_GET_HOST_CACHE_STATE").hosts:
            raise AssertionError("Fresh key unexpectedly matched an event host")
        call("REMOTE_OPERATION_REMOVE_CACHE")
        try:
            call("REMOTE_OPERATION_MATCH_LOCATION_LEN", query_type=999)
        except grpc.RpcError as error:
            # LocalRpcServer currently maps executeFunction(false) to INTERNAL.
            if error.code() != grpc.StatusCode.INTERNAL:
                raise
        else:
            raise AssertionError("RTP swallowed the invalid query type")


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
