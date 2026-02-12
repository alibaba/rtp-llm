import os
import shutil
import subprocess
from copy import deepcopy

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute


def main():
    current_dir = os.getcwd()
    current_env = dict(os.environ)
    #   --test_env=START_PORT=51234 \
    #   --test_env='CUDA_VISIBLE_DEVICES=0,1' \
    #   --test_env=TP_SIZE=1 \
    #   --test_env=WORLD_SIZE=1 \
    current_env["TP_SIZE"] = "1"
    current_env["WORLD_SIZE"] = "1"
    prefill_env = deepcopy(current_env)
    decode_env = deepcopy(current_env)
    prefill_port = "31234"
    decode_port = "32222"

    decode_endpoint = EndPoint(
        type="Vipserver",
        address=f"127.0.0.1:{decode_port}",
        protocol="http",
        path="/",
    )
    prefill_endpoint = EndPoint(
        type="Vipserver",
        address=f"127.0.0.1:{prefill_port}",
        protocol="http",
        path="/",
    )
    group_endpoint = GroupEndPoint(
        group="default",
        prefill_endpoint=prefill_endpoint,
        decode_endpoint=decode_endpoint,
    )
    service_route = ServiceRoute(
        service_id="test", role_endpoints=[group_endpoint], use_local=True
    )

    prefill_env["START_PORT"] = prefill_port
    prefill_env["CUDA_VISIBLE_DEVICES"] = "2"
    prefill_env["ROLE_TYPE"] = "PREFILL"
    prefill_env["REMOTE_RPC_SERVER_IP"] = "localhost"
    prefill_env["REMOTE_SERVER_PORT"] = decode_port
    prefill_env["MODEL_SERVICE_CONFIG"] = service_route.model_dump_json()

    decode_env["START_PORT"] = decode_port
    decode_env["CUDA_VISIBLE_DEVICES"] = "3"
    decode_env["ROLE_TYPE"] = "DECODE"
    decode_env["REMOTE_RPC_SERVER_IP"] = "localhost"
    decode_env["REMOTE_SERVER_PORT"] = prefill_port

    decode_dir = current_dir + "/decode"
    prefill_dir = current_dir + "/prefill"
    if os.path.exists(decode_dir) and os.path.isdir(decode_dir):
        shutil.rmtree(decode_dir)
    os.mkdir(decode_dir)
    if os.path.exists(prefill_dir) and os.path.isdir(prefill_dir):
        shutil.rmtree(prefill_dir)
    os.mkdir(prefill_dir)

    decode_p = subprocess.Popen(
        ["/opt/conda310/bin/python", "-m", "rtp_llm.start_server"],
        env=decode_env,
        cwd=decode_dir,
    )
    prefill_p = subprocess.Popen(
        ["/opt/conda310/bin/python", "-m", "rtp_llm.start_server"],
        env=prefill_env,
        cwd=prefill_dir,
    )
    decode_p.wait()
    prefill_p.wait()


if __name__ == "__main__":
    main()
