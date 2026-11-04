#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

target_ip="${NAVI_MASTER_IP:-215.1.142.4}"
container="${NAVI_MASTER_CONTAINER:-navi-sched-master}"
remote_user="${NAVI_MASTER_USER:-admin}"
auth_profile="${WT_AUTH_PROFILE:-default}"
connect_timeout="${NAVI_MASTER_CONNECT_TIMEOUT:-30}"
local_file="${SCRIPT_DIR}/update_scheduler_config.sh"
remote_path="${NAVI_MASTER_REMOTE_PATH:-/home/admin/hippo/worker/slave/whale_prod_Qwen3-235B-A22B-Instruct-2507_for_omega_64K_Qwen3-235B-A22B-Instruct-2507_for_omega_64K_cn_MI308X_2TP_spot.navi-sched-master_part_54_85/navi_sched_master/update_scheduler_config.sh}"
wt_bin="${WT_BIN:-${HOME}/.local/bin/wt}"

usage() {
    cat <<'EOF'
Usage: copy_to_navi_master.sh [options]

Uploads a file through the authenticated StarAgent WebTerminal SCP channel,
then downloads it again and verifies that its contents are identical.

Options:
  --target-ip IP       Master host IP
  --container NAME     Target container (default: navi-sched-master)
  --user USER          Remote user (default: admin)
  --local PATH         Local file to upload
  --remote PATH        Absolute destination path
  --timeout SECONDS    SSH connection timeout
  -h, --help           Show this help

Environment overrides:
  NAVI_MASTER_IP, NAVI_MASTER_CONTAINER, NAVI_MASTER_USER,
  NAVI_MASTER_REMOTE_PATH, NAVI_MASTER_CONNECT_TIMEOUT,
  WT_AUTH_PROFILE, WT_BIN
EOF
}

while (($# > 0)); do
    case "$1" in
        --target-ip)
            target_ip="${2:?--target-ip requires a value}"
            shift 2
            ;;
        --container)
            container="${2:?--container requires a value}"
            shift 2
            ;;
        --user)
            remote_user="${2:?--user requires a value}"
            shift 2
            ;;
        --local)
            local_file="${2:?--local requires a value}"
            shift 2
            ;;
        --remote)
            remote_path="${2:?--remote requires a value}"
            shift 2
            ;;
        --timeout)
            connect_timeout="${2:?--timeout requires a value}"
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -f "$local_file" ]] || {
    echo "local file does not exist: $local_file" >&2
    exit 1
}
[[ -x "$wt_bin" ]] || {
    echo "wt executable does not exist: $wt_bin" >&2
    exit 1
}
[[ "$target_ip" =~ ^[0-9A-Za-z:._-]+$ ]] || {
    echo "invalid target IP or host: $target_ip" >&2
    exit 2
}
[[ "$container" =~ ^[0-9A-Za-z._-]+$ ]] || {
    echo "invalid container name: $container" >&2
    exit 2
}
[[ "$remote_user" =~ ^[0-9A-Za-z._-]+$ ]] || {
    echo "invalid remote user: $remote_user" >&2
    exit 2
}
[[ "$remote_path" == /* ]] || {
    echo "remote path must be absolute: $remote_path" >&2
    exit 2
}

ssh_config="$(mktemp "${TMPDIR:-/tmp}/navi-master-scp-config.XXXXXX")"
verify_file="$(mktemp "${TMPDIR:-/tmp}/navi-master-copy-verify.XXXXXX")"
trap 'rm -f "$ssh_config" "$verify_file"' EXIT

alias_name="navi_master_copy"
target_url="https://sa-aliproduct.alibaba-inc.com/ops/terminal.html?source=suezops&ip=${target_ip}&activeNavItem=SessionManage"

cat >"$ssh_config" <<EOF
Host ${alias_name}
    ProxyCommand ${wt_bin} session proxy --target '${target_url}' --host ${alias_name} --auth-profile ${auth_profile} --provider staragent --container ${container}
    StrictHostKeyChecking no
    UserKnownHostsFile ${HOME}/.webterminal-cli/session/known_hosts
    RequestTTY no
    ControlMaster no
    ControlPath none
EOF
chmod 600 "$ssh_config"

scp -F "$ssh_config" -o "ConnectTimeout=${connect_timeout}" \
    "$local_file" "${remote_user}@${alias_name}:${remote_path}"
scp -F "$ssh_config" -o "ConnectTimeout=${connect_timeout}" \
    "${remote_user}@${alias_name}:${remote_path}" "$verify_file"

if ! cmp -s "$local_file" "$verify_file"; then
    echo "remote verification failed: downloaded content differs" >&2
    exit 1
fi

checksum="$(sha256sum "$local_file" | awk '{print $1}')"
printf 'COPY_DONE target=%s remote=%s sha256=%s\n' \
    "$target_ip" "$remote_path" "$checksum"
