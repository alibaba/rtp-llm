"""Partial HA resource fixtures without Java startup."""

from cases.master_ha_failover.client import HaReplayClient


def client_resource(**attributes):
    client = HaReplayClient.__new__(HaReplayClient)
    client.finished = False
    client.__dict__.update(attributes)
    return client


def replay_source():
    return dict(kind="trace", model="prefix_lineage", version="2", parameters=dict(priority=50))


def replay_trace(path, *_args, **_kwargs):
    import json
    path.write_text("".join(json.dumps(dict(ts=t, il=16, ol=4, bh=[7], priority=50)) + "\n"
                            for t in (0, 120000)))
    return path
