"""Write the exact qualified training-library digest map, without binaries."""
import hashlib
import json
from pathlib import Path
import sys
from q36_nonshared_launch_contract import CUDNN_SHA


if len(sys.argv) != 2:
    raise SystemExit('usage: write_cudnn_manifest.py OUTPUT')
data = (json.dumps(CUDNN_SHA, sort_keys=True, indent=2) + '\n').encode()
with Path(sys.argv[1]).open('xb') as stream:
    stream.write(data)
print(hashlib.sha256(data).hexdigest())
