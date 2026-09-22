"""Hash-pinned historical Master adapter for controlled scenario A/B tests."""

import hashlib
import json
import os
import re
from pathlib import Path


def load_manifest():
    name = os.environ.get("FLEXLB_FT_HISTORICAL_MASTER_MANIFEST")
    if not name:
        return None
    manifest = json.loads(Path(name).read_text(encoding="utf-8"))
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"]):
        raise ValueError("historical master requires a full source commit")
    for field in ("jar", "config"):
        path = Path(manifest[field])
        if not path.is_absolute():
            raise ValueError("historical master paths must be absolute")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != manifest[field + "_sha256"]:
            raise ValueError("historical master checksum mismatch: " + field)
    config = json.loads(Path(manifest["config"]).read_text(encoding="utf-8"))
    if config.get("schemaVersion") != 1 or config["dispatcher"]["type"] != "NON_BATCH":
        raise ValueError("historical adapter requires schema-1 NON_BATCH")
    return manifest


MANIFEST = load_manifest()


def adapt_env(env, menv):
    if MANIFEST is None:
        return menv
    if env.spec.discovery != "discovery_file" or env.spec.raw_config is not None:
        raise ValueError(
            "historical master requires normal dynamic file-discovery scenario"
        )
    menv["FLEXLB_CONFIG"] = Path(MANIFEST["config"]).read_text(encoding="utf-8")
    service = json.loads(menv["MODEL_SERVICE_CONFIG"])
    service.pop("discovery_file", None)
    menv["MODEL_SERVICE_CONFIG"] = json.dumps(service)
    menv["MOCK_DISCOVERY_FILE"] = str(env.discovery_file)
    (env.run_dir / "historical-master.json").write_text(
        json.dumps(MANIFEST, indent=2), encoding="utf-8"
    )
    (env.run_dir / "actual-master-config.json").write_text(
        menv["FLEXLB_CONFIG"], encoding="utf-8"
    )
    return menv
