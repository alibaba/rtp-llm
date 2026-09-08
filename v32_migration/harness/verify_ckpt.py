import json
import os
import struct
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "/home/admin/models/DeepSeek-V3.2-Exp"
idx = json.load(open(f"{d}/model.safetensors.index.json"))
wm = idx["weight_map"]
by_file = {}
for k, f in wm.items():
    by_file.setdefault(f, []).append(k)

missing_files, missing_keys, truncated, total = [], [], [], 0
for f in sorted(by_file):
    p = os.path.join(d, f)
    if not os.path.exists(p):
        missing_files.append(f)
        continue
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    data_end = max(
        v["data_offsets"][1] for k, v in hdr.items() if not k.startswith("__")
    )
    want_bytes = 8 + n + data_end
    have_bytes = os.path.getsize(p)
    if have_bytes != want_bytes:
        truncated.append((f, have_bytes, want_bytes))
    for k in by_file[f]:
        if k in hdr:
            total += hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0]
        else:
            missing_keys.append((f, k))

print(f"shards in index: {len(by_file)}  missing files: {len(missing_files)}")
print(f"missing keys: {len(missing_keys)}  truncated shards: {len(truncated)}")
print(f"tensor bytes: {total / 1e9:.1f} GB")
# index total_size assumes 2 bytes/element, so it roughly doubles for fp8 weights
print(
    f"index metadata total_size: {idx['metadata']['total_size'] / 1e9:.1f} GB (informational)"
)
if missing_files[:3]:
    print("first missing:", missing_files[:3])
if truncated[:3]:
    print("first truncated:", truncated[:3])

cfg = json.load(open(f"{d}/config.json"))
print(
    f"model_type={cfg.get('model_type')} layers={cfg.get('num_hidden_layers')} "
    f"quant={(cfg.get('quantization_config') or {}).get('quant_method')} "
    f"index_topk={cfg.get('index_topk')}"
)
ok = not missing_files and not missing_keys and not truncated
print("VERDICT:", "READY" if ok else "INCOMPLETE")
sys.exit(0 if ok else 1)
