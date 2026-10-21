"""Run one CPU Mock ViT per Pod without loading the frontend/Torch runtime."""
import os
from pathlib import Path
import shlex

from whale_frontend_fixture import prepare_assets


def command(env):
    if env.get("RTP_LLM_MOCK_BUNDLE") != "1":
        raise ValueError("RTP_LLM_MOCK_BUNDLE=1 is required")
    if env.get("ROLE_TYPE") != "VIT":
        raise ValueError("ROLE_TYPE=VIT is required")
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
    return [java, *shlex.split(env.get("MOCK_VIT_JAVA_OPTS", "-Xms64m -Xmx512m")),
            "-Djava.awt.headless=true", "-cp", str(jar), "org.flexlb.mockengine.MockVitServer",
            "--http-port", str(port), "--port", str(port + 1), "--model-path", str(model),
            "--concurrency", env.get("VIT_MAX_CONCURRENT_REQUESTS", "32"),
            "--embedding-cache-bytes", env.get("MOCK_VIT_CACHE_BYTES", "0")]


if __name__ == "__main__":
    args = command(os.environ)
    os.execv(args[0], args)
