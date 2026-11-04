#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate disposable development mTLS credentials. Never use as production PKI."""
import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import ssl
import subprocess


def create(root, members, server_ip='127.0.0.1'):
    root = Path(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    ipaddress.ip_address(server_ip)
    if not members or len(set(members)) != len(members):
        raise ValueError('distinct members required')
    def run(*args):
        subprocess.run(['openssl', *args], cwd=root, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
    run('req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', 'ca.key', '-out', 'ca.pem',
        '-days', '2', '-subj', '/CN=Engram-disposable-test-CA')
    result = {}
    fingerprints = {}
    for index, member in enumerate(members):
        name = f'member-{index}'
        run('req', '-newkey', 'rsa:2048', '-nodes', '-keyout', name + '.key',
            '-out', name + '.csr', '-subj', '/CN=' + name)
        (root / (name + '.ext')).write_text('basicConstraints=CA:FALSE\nextendedKeyUsage=serverAuth,clientAuth\nsubjectAltName=IP:' + server_ip + '\n')
        run('x509', '-req', '-in', name + '.csr', '-CA', 'ca.pem', '-CAkey', 'ca.key',
            '-CAcreateserial', '-out', name + '.pem', '-days', '2', '-extfile', name + '.ext')
        cert = root / (name + '.pem')
        fingerprints[member] = hashlib.sha256(ssl.PEM_cert_to_DER_cert(cert.read_text())).hexdigest()
        result[member] = dict(ca_file=str(root / 'ca.pem'), cert_file=str(cert), key_file=str(root / (name + '.key')))
    for item in result.values():
        item['instance_cert_sha256'] = fingerprints
    for path in root.iterdir():
        path.chmod(0o600)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--member', action='append', required=True)
    parser.add_argument('--server-ip', default='127.0.0.1')
    args = parser.parse_args()
    os.umask(0o077)
    values = create(args.output.resolve(), args.member, args.server_ip)
    path = args.output / 'tls-config.json'
    path.write_text(json.dumps(values, indent=2)); path.chmod(0o600)
    print(path)
