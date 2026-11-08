# K3.1 Prefill memory

`PREFILL_CHUNK_SIZE` controls the number of new tokens the scheduler hands to
the model in one Prefill step. Native MTP uses the same input range as the
target after restoring the target hidden states and shifting the draft tokens.

`RTP_PREFILL_KV_WORKSPACE_BUDGET_GIB` defaults to 8 GiB per rank. Target
Indexer and SFA use it for Page-RR KV ingress and history reads. Native BF16
MTP MLA uses it for the history reader and KV-up projection workspace.

The KV budget covers the temporary buffers needed to process a KV block.
Query activations, output and merge states, and persistent buffers also use
device memory. Lower `PREFILL_CHUNK_SIZE` if those activations are too large.
