"""Record the Master actually launched; optional artifact selection is not a gate."""

import hashlib
import json
import os
import re
import zipfile
from pathlib import Path


def _source_commit(jar):
    declared = os.environ.get("FLEXLB_FT_MASTER_SOURCE_COMMIT")
    if declared:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", declared):
            raise ValueError("FLEXLB_FT_MASTER_SOURCE_COMMIT must be a full SHA")
        return declared.lower(), "declared"
    try:
        with zipfile.ZipFile(jar) as archive:
            manifest = archive.read("META-INF/MANIFEST.MF").decode(errors="replace")
        for line in manifest.splitlines():
            if line.lower().startswith(("git-commit:", "build-commit:", "source-commit:")):
                value = line.split(":", 1)[1].strip()
                if re.fullmatch(r"[0-9a-fA-F]{40}", value):
                    return value.lower(), "jar-manifest"
    except (OSError, KeyError, zipfile.BadZipFile):
        pass
    return None, "unavailable"


def configure_master(env, menv, jar):
    """Apply a plain config-file override and archive observed launch inputs."""
    config_file = os.environ.get("FLEXLB_FT_MASTER_CONFIG_FILE")
    if config_file:
        menv["FLEXLB_CONFIG"] = Path(config_file).read_text(encoding="utf-8")
    if "FLEXLB_CONFIG" not in menv:
        return menv
    config = json.loads(menv["FLEXLB_CONFIG"])
    if config.get("schemaVersion") == 1 and env.spec.discovery == "discovery_file":
        service = json.loads(menv["MODEL_SERVICE_CONFIG"])
        service.pop("discovery_file", None)
        menv["MODEL_SERVICE_CONFIG"] = json.dumps(service)
        menv["MOCK_DISCOVERY_FILE"] = str(env.discovery_file)
    jar = Path(jar).resolve()
    commit, origin = _source_commit(jar)
    identity = dict(
        jar=str(jar),
        jar_sha256=hashlib.sha256(jar.read_bytes()).hexdigest(),
        source_commit=commit,
        source_commit_origin=origin,
        effective_config_sha256=hashlib.sha256(menv["FLEXLB_CONFIG"].encode()).hexdigest(),
    )
    (env.run_dir / "master-artifact.json").write_text(json.dumps(identity, indent=2))
    (env.run_dir / "actual-master-config.json").write_text(menv["FLEXLB_CONFIG"])
    return menv
