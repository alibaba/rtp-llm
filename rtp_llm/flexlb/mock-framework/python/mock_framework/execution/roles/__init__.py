"""Run-owned service roles."""

ENGINE_ROLES = ("prefill", "decode", "vit")


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


def retire_instance(topology, monitoring, instance_id, directory):
    (directory / "ready.json").unlink(missing_ok=True)
    topology.retire(instance_id)
    if monitoring:
        monitoring.retire(instance_id)
