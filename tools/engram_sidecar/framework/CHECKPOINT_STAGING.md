# Prepare a real checkpoint without consumer table copies

The [recorded real-checkpoint staging result](CHECKPOINT_STAGING_RESULTS.json)
contains verified full-source shard hashes and the exact Owner/consumer sizes.
Staging success is separate from GPU lookup and full-model HTTP acceptance.

`stage_checkpoint.py` splits an existing, immutable local safetensors checkpoint
using Python 3.6 or newer and an 8 MiB streaming buffer. It does not load torch,
initialize CUDA, or modify the source. Omitting `--execute` prints a read-only
plan. Create the private destination parent on a filesystem with enough space.

```bash
python3 tools/engram_sidecar/framework/stage_checkpoint.py \
  --source /models/DeepSeek-V4.1-Flash \
  --destination /workspace/checkpoints/owner \
  --mode owner --metadata-file encoding/encoding.py \
  --expected-sha256 /workspace/official-sha256.json
```

Use `--mode consumer` with a different destination on the inference host. After
reviewing the sizes and available space, repeat with `--execute`. An interrupted
copy can resume using the same arguments plus `--execute --resume`. The tool
rechecks source snapshots and completed-file digests; it never overwrites an
existing final destination.

The optional expected-hash file maps relative filenames to 64-character SHA256
strings from a trusted checkpoint manifest. Each full copied file is hashed
while streaming and must match any supplied expected hash. Source metadata,
index/header consistency and file sizes are checked before copying.

- Owner keeps the original shards containing Engram embeddings, including their
  small projection weights, and the required configuration/tokenizer assets.
- Consumer keeps every non-embedding tensor. Mixed shards are rebuilt with new
  safetensors headers and offsets, copying only selected payload ranges. Complete
  embedding tables are absent from both actual files and the rewritten index.
- `encoding/encoding.py` is preserved from the chosen source revision; the tool
  does not silently replace it with a newer encoder.

`checkpoint_provenance.json` records source snapshots, header hashes, copied
file/range hashes and output hashes. A consumer subset hash does **not** validate
the original mixed shard's full hash. In a paired run, Owner's full copies can
validate those shards while consumer's full copies validate the other shards.
Fresh-copy hashing covers the streamed bytes; only resume re-reads a completed
destination file. Keep these scopes distinct when reporting verification.

Publication uses a private staging directory, a lock and an atomic no-replace
rename. The 26 CPU tests cover actual subset bytes/headers, source mutation,
space/path checks, SHA rejection, restart boundaries and destination races.
This preparation step alone proves neither GPU loading nor HTTP serving.
