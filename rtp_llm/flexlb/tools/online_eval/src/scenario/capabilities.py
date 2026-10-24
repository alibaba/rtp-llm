"""Case runtime services; extension names are resolved only inside their owner."""


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
    def epoch(self):
        return self._context.env_epoch

    @property
    def metrics(self):
        return self._context.monitor.client

    @property
    def directory(self):
        return self._context.artifact_dir

    def capability(self, kind, name):
        from scenario.registry import registry
        return registry().capability(kind, name, case=self.case)

    def records(self, handle, kind, deadline):
        from adapters.client_records import read_records
        return read_records(self._context.resource(handle, kind), deadline, self._context.env_epoch)

    def event(self, identity, **kwargs):
        return self._context.record_event(identity, **kwargs)

    def source_files(self):
        """Use the declared case source inventory; actions do not import program."""
        from scenario.registry import registry
        owner = registry().cases[self.case].path.parent
        return tuple(owner / name for name in self._context.instance["implementation"]["case_sources"])
