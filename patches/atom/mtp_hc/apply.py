"""Apply the optional ROCm MTP HC patch after installing the pinned ATOM wheel."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tempfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / 'manifest.json').read_text())
    patch = directory / 'mtp_hc.patch'
    if digest(patch.read_bytes()) != manifest['patch_sha256']:
        raise RuntimeError('HC patch hash mismatch')
    package = importlib.metadata.distribution('atom')
    if package.version != manifest['atom_version']:
        raise RuntimeError(f'Expected ATOM {manifest["atom_version"]}; got {package.version}')
    target = Path(package.locate_file(manifest['file']))
    original = target.read_bytes()
    if digest(original) == manifest['patched_sha256']:
        print(f'HC patch verified: {target}')
        return
    if digest(original) != manifest['source_sha256'] or args.verify_only:
        raise RuntimeError('HC source changed or patch missing; refusing to overwrite')
    # Validate the full patched result before modifying the installed module.
    with tempfile.TemporaryDirectory(prefix='atom_mtp_hc_') as temporary:
        candidate = Path(temporary) / manifest['file']
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(original)
        subprocess.run(['git', 'apply', str(patch)], cwd=temporary, check=True)
        payload = candidate.read_bytes()
        if digest(payload) != manifest['patched_sha256']:
            raise RuntimeError('HC patched output hash mismatch')
        compile(payload, str(target), 'exec')
    target.write_bytes(payload)
    print(f'HC patch installed: {target}; restart services before enabling')


if __name__ == '__main__':
    main()
