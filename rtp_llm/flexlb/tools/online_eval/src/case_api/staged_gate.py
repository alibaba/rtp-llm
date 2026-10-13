"""Project completed persisted checks; this adapter never adjudicates again."""
from dataclasses import dataclass
from pathlib import Path
import json
from case_api.extension_contracts import validate_call


@dataclass(frozen=True)
class StagedGateProjection:
    action: str
    project: object

    def __post_init__(self):
        validate_call(self.project, 3, "staged_gate.project")

    def __call__(self, directory):
        from monitoring.metric_store import MetricStore
        directory = Path(directory)
        result = json.loads((directory / "result.json").read_text())
        store = MetricStore.read(directory)
        for stage in result["stages"]:
            if stage["action"] == self.action:
                for check in stage["checks"]:
                    if check["status"] in {"PASS", "FAIL", "WARNING"}:
                        self.project(store, stage, check)
        store.save(directory)
