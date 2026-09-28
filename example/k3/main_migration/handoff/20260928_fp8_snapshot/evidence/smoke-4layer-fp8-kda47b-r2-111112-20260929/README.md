# Four-layer FP8 PD flow, 2026-09-29

This is the raw result of the KDA direct-output revision `47b1222ff64a5f206df9daa7eb2dedbaafdd560b` on L20-dev-111 (Prefill) and L20-dev-112 (Decode). Both roles used TP8/EP8, FP8 target weights and KV cache, BF16 NCCL, and Native MTP. The target and MTP weights were loaded with FastSafetensors directly from the verified 3FS paths.

`result.json` reports 11/11 flow cases passed. `independent-flow-audit.json` independently checked the saved responses and reports 11/11 passed, no replacement characters, and observed MTP draft rounds. The `requests/` directory contains the per-case raw responses; `token-fixtures/` contains the inputs needed to inspect the cases.

This four-layer flow checks PD state handoff and MTP execution. It is not a semantic answer audit and does not establish full 93-layer smoke or 64K steady-state performance. Those remain separate acceptance gates.
