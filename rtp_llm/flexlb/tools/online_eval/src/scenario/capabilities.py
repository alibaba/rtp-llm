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
        return self._context.plan.implementation.to_dict()["configuration"]["case"]

    def source_files(self):
        """Use the declared case source inventory; actions do not import program."""
        from scenario.registry import registry

        owner = registry().cases[self.case].path.parent
        return tuple(owner / name for name in self._context.plan.implementation.to_dict()["case_sources"])
