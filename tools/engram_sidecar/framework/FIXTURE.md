# Framework Engram fixture

`checkpoint_fixture.py` creates an actual WordLevel tokenizer and two safetensors
checkpoints for the real RTP Engram loading hook. It is a complete **Engram
component** fixture; it is not a full LLM checkpoint or a service test.

Run inside the existing CUDA 13 development venv, from the main-based checkout:

```bash
# Activate the validated CUDA 13 development venv, then run from the repo root.
python tools/engram_sidecar/framework/checkpoint_fixture.py \
  --root /absolute/private/engram-framework-fixture-v2
```

The destination must be a new absolute directory. The generator uses CPU only;
it does not import native engine ops, allocate CUDA memory, or run inference.
The command reloads both saved tokenizers through `NgramHashState`, checks their
hash ABI matches, and verifies serialized tensor dtype/shape on the CPU.

| Output | Content |
|---|---|
| `owner_checkpoint/` | Two complete Engram layers: 12 weight tensors, tokenizer and index |
| `consumer_checkpoint/` | WKV/q/k only: 8 weight tensors, tokenizer and index; no embedding tables |
| `config.json` | Consumer config returned by `generate_fixture(root)` |
| `fixture.json` | Geometry, hash ABI, checkpoint locations, cases and file SHA256 |
| `oracle_inputs.safetensors` | Deterministic CPU hidden/windows/mask/hash IDs for six cases |

Layers are `0:4097` and `1:514` rows, with 64 columns and group-32 E8M0 scale.
There are six hash columns (`max_ngram_size=4`, `n_heads=2`), so WKV has K=384;
hidden is `[M, 2, 128]` and WKV N=384. Weight tensors use actual E4M3 FP8,
scales actual E8M0 FP8, and q/k BF16.

Revision `engram-framework-fixture-main-492ecf45-v2` encodes the layer slot and
row ID as FP8 `-1/+1` signs in the first 14 columns. Every one of the 4,611
actual rows has distinct weight/scale bytes, and remains distinct after
dequantization because E8M0 scales are positive. This removes the v1 fixture's
34-row repetition. Other columns retain the deterministic varying values;
dequantized values remain finite and bounded by 4 in magnitude.

API:

```python
from pathlib import Path
from tools.engram_sidecar.framework.checkpoint_fixture import generate_fixture, make_inputs

config = generate_fixture(Path('/absolute/private/engram-framework-fixture-v2'))
# config['checkpoint_path'] points to consumer_checkpoint.
# Owner loading must explicitly use root / 'owner_checkpoint'.
inputs = make_inputs(batch_size=8, step=2)
```

Saved cases are `decode0`, `decode1`, `decode7`, `mixed8`, `prefill33`, and `empty`.
The singleton decode cases use one reproducible sequence, with token at position
`t` equal to `5 + (5*t % 27)`. Windows are newest token first; steps 0, 1 and 7
contain 0, 1 and 3 valid history tokens respectively. Advancing a step shifts
the previous window into the history. Multi-row cases retain explicit sequence
boundaries, current image tokens and fully masked rows.
Keys use `case.hidden`, `case.engram_token_windows`, `case.token_mask` and
`case.hash_ids`. The hash IDs come from real CPU `NgramHashState`, including
sequence boundaries and image masking. These are inputs/hash expectations;
they are not evidence that GPU projection, model forward, or HTTP serving passed.

When transferring a consumer checkpoint to another root, set the runtime
`checkpoint_path` to its actual location. The tokenizer/hash ABI is independent
of that directory path. Keep the owner checkpoint on the owner host only; this
allows missing remote bindings or a fallback table load to fail visibly.

Run the fixture coverage tests with CPU Torch:

```bash
python tools/engram_sidecar/framework/test_checkpoint_fixture.py -v
```

They check rolling decode history, masks, all-row uniqueness before and after
dequantization, adjacent/34-row differences, and selected/empty row generation.
The fixture is still `[M, 2, 128]`: it does not exercise the production
`[M, 4, 5120]` fused gate or its in-place update path. Existing v1 checkpoints
must not be reused with the v2 oracle; generate a fresh Owner checkpoint and
transfer only the consumer subset.

Generate the complete v2 fixture **on the Owner host only**. Transfer only
`consumer_checkpoint/`, `config.json`, `fixture.json` and
`oracle_inputs.safetensors` through an authenticated channel to the consumer.
Use mode-0700 directories and mode-0600 files. Never transfer `owner_checkpoint/`
or embedding files. If the destination root differs, the consumer harness rebinds
its runtime checkpoint path without rewriting the stored configuration. Fixture
metadata also describes Owner files; consumer checks validate only the declared
files required on that consumer.
