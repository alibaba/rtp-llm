"""Run-owned service roles."""


def register_ready_instance(
    topology, monitoring, records, identity, control, endpoint, *, pid=None
):
    topology.register(identity.instance_id, identity.role, control, endpoint)
    if pid is not None:
        records.emit(
            event="instance_started",
            node=identity.instance_id,
            pid=pid,
            rpc=endpoint["rpc"],
            **identity.event_fields(),
        )
    if monitoring:
        monitoring.register_target(identity.instance_id, endpoint)
