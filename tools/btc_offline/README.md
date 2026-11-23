# BTC offline prefill replay

Build the replay program in a DSv4.1 build environment:

```sh
bazel build --config=cuda13 //tools/btc_offline:replay
```

The target uses the BTC implementation in this checkout. The metadata-only build
configuration excludes the model execution library. The current BTC library still
has native transfer dependencies. At replay time an injected submitter validates
source and target block IDs, reports buffer safety, and completes the native transfer
callback without copying KV data. The pools keep allocation and reference metadata
but allocate no KV payload. Model execution and decode are outside this prefill driver.

The source marker in `btc-offline-backend.json` identifies this target and its
three validation binaries for the replay skill. Use the corresponding RTP-LLM
runtime libraries and run without GPU devices. Validate pool accounting, native
watermarks, allocation failure, and in-process pause/continue consistency before
comparing hit rates. Checkpoint save/restore is unsupported because it couples the
driver to private BTC tree, pool, and eviction state. Keep the replay process alive
for continuation. See the skill for --pause-after and the PAUSED/CONTINUE protocol.
