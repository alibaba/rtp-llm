"""Apply only the retained private PD fixes to an exactly identified source tree."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply(root):
    root=Path(root).resolve(strict=True)
    directory=Path(__file__).resolve().parent
    source=json.loads((directory/'SOURCE.json').read_text())
    # A final version can legitimately be past an earlier step's intermediate
    # version. Check the complete final state before deciding which step to run.
    final={}
    for step in source['steps']:
        final.update(step['after'])
    def matches(expected):
        for name,value in expected.items():
            path=(root/name).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError:
                raise ValueError('private source escapes the selected repository')
            if digest(path)!=value:
                return False
        return True
    if matches(final):
        return {'status':'already_applied','files':final}
    for step in source['steps']:
        patch=directory/step['patch']
        if digest(patch)!=step['patch_sha256']:
            raise ValueError('PD patch checksum mismatch')
        if matches(step['after']):
            continue
        if not matches(step['before']):
            raise ValueError('private PD source differs; review adaptation before applying '+step['patch'])
        command=['patch','--batch','--forward','-p1','--input',str(patch)]
        subprocess.run(command+['--dry-run'],cwd=root,check=True)
        subprocess.run(command,cwd=root,check=True)
        if not matches(step['after']):
            raise RuntimeError('PD patch result checksum mismatch')
    if not matches(final):
        raise RuntimeError('private PD source did not reach the final contract')
    return {'status':'applied','files':final}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True)
    args=parser.parse_args()
    print(json.dumps(apply(args.source_root),indent=2))
