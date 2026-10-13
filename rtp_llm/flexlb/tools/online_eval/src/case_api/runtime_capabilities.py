"""Case runtime services; extension names are resolved only inside their owner."""
from copy import deepcopy


class EnvironmentServices:
    def __init__(self, context):
        self._context = context

    @property
    def current(self):
        return self._context.env

    def master_instance_target(self, environment, name):
        if environment is not self.current:
            raise ValueError("environment is outside the current epoch")
        return self._context.backend.manager.master_instance_target(environment, name)

    def client_operations(self, jvm_xms, jvm_xmx):
        from adapters.java_client import ClientOps
        return ClientOps(self, jvm_xms, jvm_xmx)


class RuntimeCapabilities:
    def __init__(self, context):
        self._context = context
        self.environment = EnvironmentServices(context)

    @property
    def case(self):
        return self._context.instance["implementation"]["configuration"]["case"]

    @property
    def directory(self):
        return self._context.artifact_dir

    def capability(self, kind, name):
        from case_api.registry import registry
        return registry().capability(kind, name, case=self.case)

    def gate(self, name):
        return self.capability("gate", name)

    def client(self, name, params, deadline):
        """The factory owns semantics; the framework owns handle and cleanup."""
        deadline.check()
        capability = self.capability("client", name)
        client = capability.factory(self, deepcopy(params), deadline)
        if not callable(getattr(client, "cleanup", None)):
            raise TypeError("registered client must implement cleanup(deadline)")
        handle = self._context.register_resource("client", client, client.cleanup, evidence=capability.evidence)
        deadline.check()
        return handle

    def snapshot(self, name, params):
        reader = self.capability("snapshot", name).factory(self, deepcopy(params))
        if not callable(getattr(reader, "read", None)):
            raise TypeError("snapshot source must implement read(deadline)")
        if callable(getattr(reader, "close", None)):
            self._context.add_cleanup("snapshot-" + name, reader.close)
        return reader

    def records(self, handle, kind, deadline):
        from case_api.client_records import read_records
        return read_records(self._context.resource(handle, kind), deadline, self._context.env_epoch)

    def event(self, identity, **kwargs):
        return self._context.record_event(identity, **kwargs)

    def source_files(self):
        """Use the declared case source inventory; actions do not import program."""
        from case_api.registry import registry
        owner = registry().cases[self.case].path.parent
        return tuple(owner / name for name in self._context.instance["implementation"]["case_sources"])
