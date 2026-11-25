"""Apply one fault on the launcher-owned PACE stack until the test releases it."""

import json
import os
import socket
import subprocess
import sys
from contextlib import contextmanager


@contextmanager
def tcp_fault(scenario, device, consumer_ip, provider_ip, tent_port, tc_binary):
    netem = {
        "slow-read": ("delay", "3s"),
        "slow-write": ("delay", "3s"),
        "packet-loss": ("loss", "100%"),
    }[scenario]

    def tc(*args):
        return subprocess.check_output([tc_binary, *args], text=True, timeout=5)

    def hits():
        qdiscs = json.loads(tc("-j", "-s", "qdisc", "show", "dev", device))
        stats = next(qdisc for qdisc in qdiscs if qdisc["kind"] == "netem" and qdisc["handle"] == "20:")
        if scenario == "packet-loss":
            return stats["drops"]
        return stats["packets"] + stats["qlen"]

    tc("qdisc", "add", "dev", device, "root", "handle", "1:", "prio",
       "bands", "2", "priomap", *["0"] * 16)
    try:
        tc("qdisc", "add", "dev", device, "parent", "1:2", "handle", "20:", "netem", *netem)
        tc("filter", "add", "dev", device, "parent", "1:", "protocol", "ip", "prio", "1",
           "flower", "skip_hw", "ip_proto", "tcp", "src_ip", consumer_ip,
           "dst_ip", provider_ip, "dst_port", tent_port, "classid", "1:2")
        yield hits
    finally:
        tc("qdisc", "del", "dev", device, "root", "handle", "1:")


def serve(path, uid, gid, device, consumer_ip, provider_ip, tent_port, tc_binary="tc"):
    with socket.socket(socket.AF_UNIX) as server:
        os.umask(0o177)
        server.bind(path)
        os.chown(path, int(uid), int(gid))
        server.listen(1)
        while True:
            peer, _ = server.accept()
            with peer, peer.makefile("rb") as requests:
                peer.settimeout(25)
                scenario = ""
                try:
                    scenario = requests.readline(64).strip().decode("ascii")
                    if not scenario:
                        continue
                    with tcp_fault(scenario, device, consumer_ip, provider_ip, tent_port, tc_binary) as hits:
                        peer.sendall(b"1")
                        if requests.readline(2) != b"\n":
                            continue
                        count = hits()
                    if count <= 0:
                        raise RuntimeError("No traffic hit the fault")
                    print(f"{scenario}: hits={count}, fault removed")
                    peer.sendall(b"1")
                except (OSError, ValueError, RuntimeError, LookupError, StopIteration,
                        subprocess.SubprocessError) as error:
                    print(f"{scenario}: {error}", file=sys.stderr)
                    try:
                        peer.sendall(b"0")
                    except OSError:
                        pass


if __name__ == "__main__":
    serve(*sys.argv[1:])
