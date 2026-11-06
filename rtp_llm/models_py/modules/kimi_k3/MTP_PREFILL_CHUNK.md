# Native K3.1 MTP prefill

`KIMI_K3_MTP_PREFILL_CHUNK_TOKENS` limits the packed native MTP query rows in a round. It defaults to 8192; set it to 0 for the previous full-sequence path. The positive budget must be a multiple of TP size. The budget is shared across requests.

The executor restores the target CP hidden states in their existing natural order, then slices the already shifted draft tokens and matching target features. Each round advances the original request's prefix and uses its existing Page-RR cache table. Every prompt query writes its KV once. Terminal recurrent H is selected before its gather and normalization; only normalized terminal rows enter the logits head. Intermediate rounds return empty hidden outputs and release their consumed attention plans. Target CP and Decode/CUDA Graph execution stay on their existing paths.

This route currently applies to TP>1, EP=TP, DP1 and the native global-MLA handoff from a CP target, without PD, multimodal features, or engram windows. Those inputs use the previous path. A full target-hidden restoration is still live while draft chunks run.

Prefill Indexer, SFA, and native BF16 MLA each use an 8 GiB per-rank KV workspace budget by default. Set `RTP_PREFILL_KV_WORKSPACE_BUDGET_GIB` to a positive GiB value to override it. The planner charges Indexer score storage and selector scratch, Page-RR pack/gather/restore, and the BF16 MLA expanded and latent buffers when they overlap with the reader. BF16 MLA uses packed skip-head-mid KV-up for history under this budget.

The budget applies to the temporary buffers used to process a KV block. Query tensors, output/merge states, persistent workspaces, and request metadata are accounted for separately in total memory. Later query rounds read earlier KV and may repeat KV-up projection.

Changing query geometry can change BF16 rounding and kernel selection. Check complete input coverage, terminal recurrent H, logits, valid KV writes, and generated tokens when comparing against budget 0. Collect allocator histories separately from tensor-value diagnostics.
