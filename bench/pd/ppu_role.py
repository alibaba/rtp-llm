#!/usr/bin/env python3
"""Start one PPU PD role from an env-style JSON config (key -> value, exported as env for rtp_llm.start_server).

  ppu_role.py <out_config.json> --base <base_config.json> [--role decode|prefill] [--set K=V ...]
              [--decode-addrs ip:port,... [--service-id ID]] [--mtp N] [--barrier N] [--force-accept X] [--rdma 0|1] [--launch RUN_DIR]

Writes the merged config and, with --launch, starts the role detached in its own session (pid in
RUN_DIR/<role>.pid, stdout in RUN_DIR/<role>.stdout.log) after checking that its port range is free.
Base configs are the deployment's own (paths, model, PPU SDK env); this script only applies the PD knobs
used by the hetero-PD measurements:
  --decode-addrs   prefill: every decode DP rank's http address (MODEL_SERVICE_CONFIG decode_endpoint;
                   --service-id renames the service)
  --mtp N          MTP depth (SP_TYPE=mtp, GEN_NUM_PER_CYCLE=N); 0 = no speculation
  --barrier N      decode: DSV4_DECODE_BARRIER_BATCH=N (hold decode until N streams are ready; 0 = off)
  --force-accept X decode: DSV4_SP_FORCE_ACCEPT_LEN=X (0 = real acceptance)
  --rdma 0|1       CACHE_STORE_RDMA_MODE
A decode that runs MTP behind an MTP prefill needs DSV4_PD_SKIP_MTP_MODULE_LOAD=1 (set it in the base or --set).
"""
import argparse
import json
import os
import socket
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--base", required=True)
    ap.add_argument("--role", choices=("decode", "prefill"))
    ap.add_argument("--set", action="append", default=[], metavar="K=V")
    ap.add_argument("--decode-addrs", default="")
    ap.add_argument("--mtp", type=int)
    ap.add_argument("--barrier", type=int)
    ap.add_argument("--force-accept")
    ap.add_argument("--rdma", choices=("0", "1"))
    ap.add_argument("--service-id")
    ap.add_argument("--launch", metavar="RUN_DIR")
    a = ap.parse_args()

    c = json.load(open(a.base))
    role = a.role or c["ROLE_TYPE"].lower()
    c["ROLE_TYPE"] = role.upper()
    if a.mtp is not None:
        if a.mtp > 0:
            c.update(
                SP_TYPE="mtp",
                SP_MODEL_TYPE="deepseek_v4_mtp",
                SP_CHECKPOINT_PATH=c["CHECKPOINT_PATH"],
                GEN_NUM_PER_CIRCLE=str(a.mtp),
                GEN_NUM_PER_CYCLE=str(a.mtp),
            )
        else:
            c.update(SP_TYPE="", SP_MODEL_TYPE="", GEN_NUM_PER_CIRCLE="0")
            c.pop("GEN_NUM_PER_CYCLE", None)
    if a.barrier is not None:
        c["DSV4_DECODE_BARRIER_BATCH"] = str(a.barrier)
    if a.force_accept is not None:
        c["DSV4_SP_FORCE_ACCEPT_LEN"] = a.force_accept
    if a.rdma is not None:
        c["CACHE_STORE_RDMA_MODE"] = int(a.rdma)
    if a.decode_addrs:
        for k in [k for k in c if k.startswith("DOMAIN_ADDRESS:")]:
            del c[k]
        msc = json.loads(c.get("MODEL_SERVICE_CONFIG") or '{"use_local": true}')
        msc["role_endpoints"] = [
            {
                "group": "default",
                "decode_endpoint": {
                    "type": "domain",
                    "address": a.decode_addrs,
                    "protocol": "http",
                    "path": "/",
                },
            }
        ]
        if a.service_id:
            msc["service_id"] = a.service_id
        c["MODEL_SERVICE_CONFIG"] = json.dumps(msc)
        c["DOMAIN_ADDRESS:" + a.decode_addrs] = a.decode_addrs
    for kv in a.set:
        k, v = kv.split("=", 1)
        c[k] = v
    json.dump(c, open(a.out, "w"), indent=2)
    print(
        json.dumps(
            {
                k: c.get(k)
                for k in (
                    "ROLE_TYPE",
                    "START_PORT",
                    "CUDA_VISIBLE_DEVICES",
                    "SP_TYPE",
                    "GEN_NUM_PER_CIRCLE",
                    "CACHE_STORE_RDMA_MODE",
                    "DSV4_DECODE_BARRIER_BATCH",
                    "DSV4_SP_FORCE_ACCEPT_LEN",
                )
            }
        )
    )
    if not a.launch:
        return
    run = a.launch
    os.makedirs(run, exist_ok=True)
    os.makedirs(c["LOG_PATH"], exist_ok=True)
    port, world = int(c["START_PORT"]), int(c["LOCAL_WORLD_SIZE"])
    for p in range(port - 10, port + world * int(c.get("WORKER_INFO_PORT_NUM", 9))):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", p))
            except OSError as e:
                sys.exit("port %d unavailable: %s" % (p, e))
    env = dict(os.environ)
    env.update({k: str(v) for k, v in c.items()})
    with open(os.path.join(run, role + ".stdout.log"), "ab", buffering=0) as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "rtp_llm.start_server"],
            cwd=run,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    open(os.path.join(run, role + ".pid"), "w").write("%d\n" % proc.pid)
    print(json.dumps({"role": role, "pid": proc.pid, "run": run}))


if __name__ == "__main__":
    main()
