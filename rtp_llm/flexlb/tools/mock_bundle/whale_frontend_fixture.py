"""Start main frontend or CPU ViT with real DSV4.1 checkpoint metadata.

The historical filename is the CI packaging contract. No synthetic tokenizer,
model config, native symbol replacement or model weights are generated here.
"""

import concurrent.futures
import hashlib
import json
import os
import sys
from pathlib import Path


def prepare_assets(source, destination):
    if not source.startswith("oss://"):
        path = Path(source)
        if not (path / "config.json").is_file():
            raise ValueError("model path lacks config.json")
        return path
    import base64
    import email.utils
    import hmac
    import urllib.error
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    url = urllib.parse.urlsplit(source)
    query = dict(urllib.parse.parse_qsl(url.query))
    # Keep the deployment's regional endpoint. Credentials never enter artifacts/logs.
    endpoint = query["OSS_ENDPOINT"]
    prefix = url.path.strip("/")
    destination.mkdir(parents=True, exist_ok=True)

    def get(key, listing=False):
        date = email.utils.formatdate(usegmt=True)
        resource = "/" + url.netloc + "/" + ("" if listing else key)
        signature = base64.b64encode(
            hmac.new(query["OSS_ACCESS_KEY"].encode(), ("GET\n\n\n" + date + "\n" + resource).encode(), hashlib.sha1).digest()
        ).decode()
        path = "/?" + urllib.parse.urlencode({"prefix": key, "max-keys": 1000}) if listing else "/" + urllib.parse.quote(key, safe="/")
        request = urllib.request.Request(
            "https://" + url.netloc + "." + endpoint + path,
            headers={"Date": date, "Authorization": "OSS " + query["OSS_ACCESS_ID"] + ":" + signature},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(64 * 1024 * 1024 + 1)
                if len(data) > 64 * 1024 * 1024:
                    raise ValueError("model metadata file exceeds byte budget")
                return data
        except urllib.error.HTTPError as error:
            if error.code == 404 and not listing:
                return None
            raise RuntimeError("model metadata fetch HTTP " + str(error.code)) from None

    files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "processor_config.json",
        "preprocessor_config.json",
    ]
    listing = ET.fromstring(get(prefix + "/encoding/", True))
    if listing.findtext("IsTruncated", "false") == "true":
        raise ValueError("encoding listing truncated")
    for item in listing.findall("Contents"):
        key, size = item.findtext("Key"), int(item.findtext("Size", "0"))
        if key and 0 < size <= 64 * 1024 * 1024:
            files.append(key[len(prefix) + 1 :])

    def fetch(name):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid metadata path")
        data = get(prefix + "/" + name)
        if data is not None:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch, files))
    if not (destination / "config.json").is_file():
        raise ValueError("checkpoint config missing")
    manifest = {
        str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.rglob("*")
        if p.is_file() and p.name != "asset-sha256.json"
    }
    (destination / "asset-sha256.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))
    return destination


def start_mock_vit(model):
    """One independent Java ViT child; HTTP health checks its actual gRPC status."""
    import signal
    import subprocess
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import grpc
    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import MultimodalRpcServiceStub

    from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb

    java = str(Path(os.environ.get("JAVA_HOME", "/opt/taobao/java")) / "bin/java")
    jar = Path(os.environ.get("MOCK_BUNDLE_JAR_DIR", "/opt/flexlb-mock/tools/mock_bundle/jars")) / "mock.jar"
    port = int(os.environ.get("MOCK_VIT_PORT", "22161"))
    from runtime_configs import calibration_input, engine_input

    runtime = Path(os.environ.get("HIPPO_PROC_WORKDIR", "/tmp")) / "mock-vit-config"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "engine-config.json").write_text(json.dumps(engine_input(os.environ)))
    (runtime / "calibration.json").write_text(json.dumps(calibration_input(os.environ)))
    args = [
        java,
        "-Xms64m",
        "-Xmx512m",
        "-Djava.awt.headless=true",
        "-cp",
        str(jar),
        "org.flexlb.mockengine.MockVitServer",
        "--mock.port",
        str(port),
        "--mock.model_path",
        str(model),
        "--mock.engine_config",
        str(runtime / "engine-config.json"),
        "--mock.calibration",
        str(runtime / "calibration.json"),
    ]
    child = subprocess.Popen(args)
    channel = grpc.insecure_channel("127.0.0.1:" + str(port))
    stub = MultimodalRpcServiceStub(channel)

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            ok = False
            if self.path in ("/", "/health") and child.poll() is None:
                try:
                    status = stub.GetWorkerStatus(pb.StatusVersionPB(), timeout=1)
                    ok = status.alive and status.role == "VIT"
                except grpc.RpcError:
                    pass
            self.send_response(200 if ok else 503)
            self.end_headers()
            self.wfile.write(b"ok" if ok else b"not ready")

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("START_PORT", "22160"))), Health)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: child.send_signal(signum))
    try:
        code = child.wait()
        if code:
            raise SystemExit(code)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        http.shutdown()
        http.server_close()
        thread.join(timeout=3)
        channel.close()


def main():
    if os.environ.get("RTP_LLM_MOCK_FRONTEND") != "1":
        raise ValueError("test deployment requires RTP_LLM_MOCK_FRONTEND=1")
    source = Path(os.environ.get("MOCK_FRONTEND_SOURCE_ROOT", str(Path(__file__).parents[4])))
    sys.path.insert(0, str(source))
    runtime = Path(os.environ.get("HIPPO_PROC_WORKDIR", "/tmp")) / "main-vit-assets"
    checkpoint = os.environ.get("CHECKPOINT_PATH") or os.environ["TOKENIZER_PATH"]
    model = prepare_assets(checkpoint, runtime)
    tokenizer = os.environ.get("TOKENIZER_PATH") or checkpoint
    tokenizer = model if tokenizer == checkpoint else prepare_assets(tokenizer, runtime / "tokenizer")
    os.environ.update(CHECKPOINT_PATH=str(model), TOKENIZER_PATH=str(tokenizer))
    if os.environ.get("MOCK_BUNDLE_ROLE") == "vit":
        start_mock_vit(model)
        return
    # Keep the native hash implementation from the pinned main runtime.
    from rtp_llm.ops import cpp_get_block_cache_keys

    assert cpp_get_block_cache_keys([[1, 2]], 123) == [-8366447758695274051]
    # Use the real supervisor so FRONTEND_SERVER_COUNT, shared admission,
    # instance coordination and shutdown behavior are effective at runtime.
    from rtp_llm.start_server import main as frontend_main

    sys.argv = [sys.argv[0]]
    frontend_main()


if __name__ == "__main__":
    main()
