#!/usr/bin/env bash
# Run as root on the D HOST. Update only this live container's device rules,
# retaining its existing PPU/device permissions. No container restart required.
set -euo pipefail
[[ $(id -u) == 0 ]] || { echo 'run as root on the host' >&2; exit 1; }
v41_container="${1:?usage: enable_rdma_devices.sh CONTAINER}"
v41_id=$(pouch inspect "$v41_container" | python3 -c 'import json,sys; c=json.load(sys.stdin)[0]; print(c.get("Id") or c["ID"])')
v41_pid=$(pouch inspect "$v41_container" | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["State"]["Pid"])')
umask 077
v41_backup=$(mktemp -d /tmp/v41-rdma-devices.XXXXXX)
v41_root="${RTP_V41_RUNC_ROOT:-}"
if [[ -z "$v41_root" ]]; then
  while IFS= read -r v41_state; do
    v41_candidate=$(dirname "$(dirname "$v41_state")")
    if runc --root "$v41_candidate" state "$v41_id" > "$v41_backup/state.json" 2>/dev/null; then
      v41_root="$v41_candidate"
      break
    fi
  done < <(find /run -type f -path "*/$v41_id/state.json" 2>/dev/null)
fi
if [[ -z "$v41_root" ]]; then
  echo "cannot locate runc state on this host; set RTP_V41_RUNC_ROOT to this container's runc root" >&2
  exit 1
fi
runc --root "$v41_root" state "$v41_id" > "$v41_backup/state.json"
python3 - "$v41_backup" "$v41_pid" <<'PY'
import json
import os
import pathlib
import stat
import sys

backup = pathlib.Path(sys.argv[1])
state = json.loads((backup / "state.json").read_text())
if int(state["pid"]) != int(sys.argv[2]):
    raise RuntimeError("runc state PID differs from the Pouch container")
config = json.loads((pathlib.Path(state["bundle"]) / "config.json").read_text())
rules = config["linux"]["resources"]["devices"]
if not rules:
    raise RuntimeError("missing existing device rules; refusing to replace them")
(backup / "devices-before.json").write_text(json.dumps({"devices": rules}, indent=2))
rules = list(rules)
for path in sorted(pathlib.Path("/dev/infiniband").glob("*")):
    info = path.stat()
    if stat.S_ISCHR(info.st_mode):
        rule = {"allow": True, "type": "c", "major": os.major(info.st_rdev),
                "minor": os.minor(info.st_rdev), "access": "rwm"}
        if rule not in rules:
            rules.append(rule)
if len(rules) == len(config["linux"]["resources"]["devices"]):
    raise RuntimeError("no RDMA device rules to add; inspect the current mapping")
(backup / "devices-after.json").write_text(json.dumps({"devices": rules}, indent=2))
PY
echo "container=$v41_container id=$v41_id runc_root=$v41_root backup=$v41_backup"
runc --root "$v41_root" update --resources "$v41_backup/devices-after.json" "$v41_id"
pouch exec -u root "$v41_container" /opt/conda312/bin/python -c '
import os,pathlib
nodes=sorted(pathlib.Path("/dev/infiniband").glob("uverbs*"))
if not nodes: raise RuntimeError("RDMA nodes are missing inside the container")
for path in nodes:
    fd=os.open(path,os.O_RDWR);os.close(fd);print(str(path)+": open_ok")
'
echo 'RDMA device access enabled for this running container; reapply after container recreation/restart.'
