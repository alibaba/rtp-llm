"""Opt-in application timing; timestamps are not socket or GPU timestamps."""

import json
import logging
import os
import time


def begin(client_id, server_request_id, rank, server):
    if os.environ.get("RTP_FRONTEND_MEASUREMENT") != "1":
        return None
    return dict(
        client_request_id=str(client_id)[:128],
        server_request_id=server_request_id,
        rank=rank,
        server=server,
        handler_entered_wall=time.time(),
        serialization_cpu_s=0.0,
        serialization_wall_s=0.0,
        sse_events=0,
        sse_json_bytes=0,
    )


def serialized(measurement, cpu_start, wall_start, data):
    if measurement is None:
        return
    now = time.time()
    measurement.setdefault("first_response_wall", wall_start)
    measurement["last_response_wall"] = wall_start
    measurement["serialization_cpu_s"] += time.thread_time() - cpu_start
    measurement["serialization_wall_s"] += now - wall_start
    measurement["sse_events"] += 1
    measurement["sse_json_bytes"] += len(data.encode("utf-8"))


def finish(measurement, status):
    if measurement is None:
        return
    measurement["handler_finished_wall"] = time.time()
    measurement["status"] = status
    logging.info("FRONTEND_MEASURE %s", json.dumps(measurement, separators=(",", ":")))
