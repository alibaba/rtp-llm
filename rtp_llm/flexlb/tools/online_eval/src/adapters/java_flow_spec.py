"""Pure Java flow admission shared by case authors and stage execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING, dataclass, field, fields
import re

from input_contract import NumberRule, mapping_fields
from adapters.client_configuration import ClientConfiguration
from adapters.load_client import client_environment
from traffic.traffic_source import SourceSpecification
from traffic.contracts import source_priority


@dataclass(frozen=True)
class JavaFlowSpec:
    group_id: str
    phase_id: str
    poll_s: float = field(metadata={'number': NumberRule(False, 0)})
    source: dict
    client: dict
    jvm_xms: str
    jvm_xmx: str
    max_requests: int | None = field(default=None, metadata={'number': NumberRule(True, 1, 1000000)})

    @classmethod
    def input_fields(cls, *, max_requests=False):
        return {item.name: item.metadata.get('number') for item in fields(cls)
                if item.name != 'max_requests' or max_requests}

    @classmethod
    def read(cls, value):
        definitions = cls.input_fields(max_requests=True)
        required = {item.name for item in fields(cls)
                    if item.default is MISSING and item.default_factory is MISSING}
        mapping_fields(value, definitions, 'java_flow', required=required)
        if value.get("max_requests", 1) is None:
            raise ValueError("max_requests must be a positive bounded request count")
        spec = cls(**deepcopy(value))
        for item in fields(spec):
            rule = item.metadata.get('number')
            value = getattr(spec, item.name)
            if rule is not None and value is not None:
                rule.validate(value, 'java_flow.' + item.name)
        if not all(isinstance(value, str) and value for value in
                   (spec.group_id, spec.phase_id, spec.jvm_xms, spec.jvm_xmx)):
            raise ValueError('flow identity and JVM sizing must be explicit')
        if re.fullmatch(r'[A-Za-z0-9_-]+', spec.group_id) is None:
            raise ValueError('group_id must be a safe directory component')
        if spec.poll_s <= 0:
            raise ValueError('invalid flow polling interval')
        SourceSpecification.read(spec.source)
        client_environment(spec.client, priority=source_priority(spec.source))
        return spec

    @property
    def configuration(self):
        return ClientConfiguration.read(self.client)
