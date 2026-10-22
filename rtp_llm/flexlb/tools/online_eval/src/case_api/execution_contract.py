"""Case/action-owned prerequisites, checked before launching any process."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionRequirements:
    request_evidence: bool = False
    monitoring: bool = False

    def __post_init__(self):
        if type(self.request_evidence) is not bool or type(self.monitoring) is not bool:
            raise ValueError('execution requirements must be boolean')

    def validate(self, execution):
        if self.request_evidence and execution.get('collection', 'request') == 'aggregate':
            raise ValueError('execution.collection requires request or diagnostic evidence')
        if self.monitoring and (not isinstance(execution.get('monitoring'), dict) or execution['monitoring'].get('capture_metrics', True) is False):
            raise ValueError('execution.monitoring.capture_metrics must be true for this consumer')
