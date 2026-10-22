"""Case 作者使用的步骤构建器与输出引用。"""

import copy
from monitoring.identity import METRIC_ID
from document_io import DocumentError as ScenarioError
from scenario.config_data import merge_data

def output(stage, field):
    """Refer to an earlier Python step's typed output; never evaluated as Python."""
    return {"$ref": f"stages.{stage}.output.{field}"}


class CaseBuilder:
    """Python owns ordering, branches and checks; YAML supplies declared parameters."""

    def __init__(self, environment, parameters, *, number_rules, profiles=(), optional_numbers=()):
        self.parameters = copy.deepcopy(parameters)
        self.number_rules = dict(number_rules)
        self.profiles = tuple(profiles)
        self.optional_numbers = frozenset(optional_numbers)
        from case_api.environment import EnvironmentView
        self.env = EnvironmentView(environment, self.profiles)
        self.steps = []
        self.read_parameters = set()
        self.metric_dependencies = {}

    def value(self, path):
        """Read required YAML data; programs provide no hidden fallback values."""
        self.read_parameters.add(path)
        value = self.parameters
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ScenarioError(f"missing YAML parameter {path!r}")
            value = value[part]
        return copy.deepcopy(value)

    def bind(self, contract):
        return contract.bind(self)

    def metric(self, identity, *, unit=None, labels=(), mode=None):
        """Declare a numeric dependency; compilation binds it to the selected plan."""
        if type(identity) is not str or not METRIC_ID.fullmatch(identity):
            raise ScenarioError("invalid metric id: " + str(identity))
        requirement = dict(unit=unit, labels=tuple(labels), mode=mode)
        if identity in self.metric_dependencies and self.metric_dependencies[identity] != requirement:
            raise ScenarioError("conflicting metric dependency: " + identity)
        self.metric_dependencies[identity] = requirement
        return identity

    def params(self, path, dynamic):
        values = self.value(path)
        if not isinstance(values, dict):
            raise ScenarioError(f"YAML parameter {path!r} must be a mapping")
        return merge_data(values, dynamic)

    def number(self, name):
        if name not in self.number_rules:
            raise ScenarioError("undeclared numeric parameter: " + name)
        try:
            return self.number_rules[name].validate(self.value(name), name)
        except ValueError as exc:
            raise ScenarioError(str(exc)) from exc

    def validate_numbers(self):
        """Apply program types and YAML narrowing without consuming business inputs."""
        reads = set(self.read_parameters)
        try:
            for name in self.number_rules:
                if name in self.optional_numbers:
                    value = self.parameters
                    for part in name.split("."):
                        if not isinstance(value, dict):
                            raise ScenarioError("invalid parameter parent: " + name)
                        if part not in value:
                            break
                        value = value[part]
                    else:
                        self.number(name)
                    continue
                self.number(name)
        finally:
            self.read_parameters = reads

    def step(self, name, action, *, params=None, timeout_s=None):
        step = {"id": name, "action": action}
        if params is not None:
            step["params"] = copy.deepcopy(params)
        if timeout_s is not None:
            step["timeout_s"] = timeout_s
        self.steps.append(step)

    def observe(self, name, action, *, params=None, timeout_s=None):
        """Independent observation: a FAIL may continue in the workload executor.

        Do not use for prerequisites, injections or checks authorizing a mutation.
        ERROR/TIMEOUT still abort dependent execution in either test class.
        """
        self.step(name, action, params=params, timeout_s=timeout_s)
        self.steps[-1]["purpose"] = "observation"

    def finish(self):
        return copy.deepcopy(self.steps)
