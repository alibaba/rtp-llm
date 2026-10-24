"""Run one CPU Mock ViT per Pod without loading the frontend/Torch runtime."""

import json
import os
import shlex
from pathlib import Path

from whale_frontend_fixture import prepare_assets


def command(env):
    if env.get("RTP_LLM_MOCK_BUNDLE") != "1":
        raise ValueError("RTP_LLM_MOCK_BUNDLE=1 is required")
    if env.get("ROLE_TYPE") != "VIT":
        raise ValueError("ROLE_TYPE=VIT is required")
    kmonitor = env.get("MOCK_KMONITOR_ENABLED", "true").lower()
    if kmonitor not in ("true", "false", "1", "0", "yes", "no"):
        raise ValueError("invalid MOCK_KMONITOR_ENABLED")
    kmonitor = "true" if kmonitor in ("true", "1", "yes") else "false"
    port = int(env.get("START_PORT", "22160"))
    if not 0 < port < 65535:
        raise ValueError("invalid ViT health port")
    source = env.get("CHECKPOINT_PATH") or env.get("TOKENIZER_PATH")
    if not source:
        raise ValueError("ViT requires real model metadata")
    runtime = Path(env.get("HIPPO_PROC_WORKDIR", "/tmp")) / "mock-vit-assets"
    model = prepare_assets(source, runtime)
    java = str(Path(env.get("JAVA_HOME", "/opt/taobao/java")) / "bin/java")
    jar = Path(env.get("MOCK_BUNDLE_JAR_DIR", str(Path(__file__).parent / "jars"))) / "mock.jar"
    from runtime_configs import calibration_input, engine_input

    engine = engine_input(env)
    calibration = calibration_input(env)
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "engine-config.json").write_text(json.dumps(engine))
    (runtime / "calibration.json").write_text(json.dumps(calibration))
    return [
        java,
        *shlex.split(env.get("MOCK_VIT_JAVA_OPTS", "-Xms64m -Xmx512m")),
        "-Djava.awt.headless=true",
        "-cp",
        str(jar),
        "org.flexlb.mockengine.MockVitServer",
        "--kmonitor",
        kmonitor,
        "--http-port",
        str(port),
        "--port",
        str(port + 1),
        "--model-path",
        str(model),
        "--engine-config",
        str(runtime / "engine-config.json"),
        "--calibration",
        str(runtime / "calibration.json"),
    ]


if __name__ == "__main__":
    args = command(os.environ)
    os.execv(args[0], args)
