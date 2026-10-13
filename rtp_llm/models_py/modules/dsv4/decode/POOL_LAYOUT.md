# DSV4 BlockPool layout (probed at runtime, 2026-05-01)

`seq_size_per_block` runtime config = 64, but framework actually allocates pools
at **256 tokens/block** for KV pools. Always derive `entries_per_block` from
`stride_bytes // bytes_per_entry`, never from the runtime config.

| attn_type id | name           | entries/block | bytes/entry | dtype                | Notes |
|--------------|----------------|---------------|-------------|----------------------|-------|
| 1            | CSA_KV         | 64            | 1024        | bf16, head_dim=512   | shared variable-length pool with 2,3 |
| 2            | HCA_KV         | 2             | 1024        | bf16, head_dim=512   | shared (same `block_id` space as 1,3) |
| 3            | INDEXER_KV     | 64            | 256         | bf16, idx_head=128   | shared |
| 4            | INDEXER_STATE  | 1 slot        | 2048        | fp32, 2*coff*idx_hd  | fixed N blocks/req; 1 compressor slot per page |
| 5            | CSA_STATE      | 1 slot        | 8192        | fp32, 2*coff*hd      | 8 blocks/req → covers coff*ratio = 8 slots |
| 6            | HCA_STATE      | 2 slots       | 4096        | fp32, 2*hd           | LOSSY: only 16 of 128 needed slots persist (existing design) |
| 7            | SWA_KV         | 256           | 1024        | bf16, head_dim=512   | 256 blocks total → 1-2 blocks/req cyclic |

## Slot mapping formulas

For multi-entry KV pools (1/2/3/7):
```
slot = block_table[req, abs_pos // entries_per_block] * entries_per_block
     + (abs_pos % entries_per_block)
```
Where `abs_pos` is:
- SWA_KV (7): token absolute position
- CSA_KV (1): compressed-K index = (token_pos + 1) / 4 - 1, only when on boundary
- HCA_KV (2): same with /128
- INDEXER_KV (3): same as CSA, /4

For state pools (4/5/6) with 1 slot/block:
```
state_slot = block_table[req, slot_in_compressor]   # direct block_id
```
Where `slot_in_compressor` ∈ [0, coff*ratio) ranges over compressor.kv_state's 2nd dim.

## DType views

```python
# KV pool view as [num_blocks * entries_per_block, head_dim] in bf16:
kv_view = pool[:, :entries_per_block * bytes_per_entry] \
            .view(torch.bfloat16) \
            .view(-1, head_dim)

# State pool view as [num_blocks * entries_per_block, 2 * coff * head_dim] in fp32
# (kv_state half | score_state half packed):
state_view = pool[:, :entries_per_block * bytes_per_entry] \
               .view(torch.float32) \
               .view(-1, 2 * coff * head_dim)
```

## Per-layer pool presence

- ratio==0 (SWA-only): only pool 7
- ratio==4 (CSA): pools 1, 3, 4, 5, 7
- ratio==128 (HCA): pools 2, 6, 7

(matches `_layer_pools` in deepseek_v4_model.py)
