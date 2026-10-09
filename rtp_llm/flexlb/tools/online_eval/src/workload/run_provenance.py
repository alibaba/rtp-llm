"""Freeze observed launch inputs before reports are assembled."""

import json
from pathlib import Path


def collect(directory, environments, metadata=None):
    epochs = {}
    for epoch, location in environments.items():
        root = Path(location)
        data = {"directory": str(root), "topology": (metadata or {}).get(epoch)}
        for field, filename in (("master_artifact", "master-artifact.json"),
                                ("actual_master_config", "actual-master-config.json"),
                                ("performance", "perf.json"), ("master_config", "master_config.json")):
            path = root / filename
            if path.is_file():
                data[field] = json.loads(path.read_text())
        epochs[epoch] = data
    flows = [json.loads(path.read_text()) for path in sorted(Path(directory).glob("**/flow-input.json"))]
    from runtime.paths import MOCK_JAR
    from traffic.traffic_source import sha256_file
    mock_artifact = dict(path=str(MOCK_JAR), sha256=sha256_file(MOCK_JAR)) if Path(MOCK_JAR).is_file() else None
    return dict(environments=epochs, flows=flows, mock_artifact=mock_artifact)
