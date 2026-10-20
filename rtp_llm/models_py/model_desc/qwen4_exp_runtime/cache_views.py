"""Strict PLE cache payload and physical page views."""

import torch

from rtp_llm.models_py.utils.cache_payload_view import (
    CachePayloadGeometry,
    payload_view,
    same_tensor_metadata,
)
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs


class PLECacheViews:
    @staticmethod
    def _fixed_state_pool(
        cache: LayerKVCache, dtype: torch.dtype, entries: int, width: int
    ) -> torch.Tensor:
        return payload_view(
            cache.kv_cache_base,
            CachePayloadGeometry(dtype, dtype, entries, width),
            f"PLE cache {cache.tag!r}",
        )

    @staticmethod
    def _resolve_block_table(
        attention_inputs: PyAttentionInputs,
        tag: str,
    ) -> torch.Tensor:
        """The request-local HybridPool physical block table for one cache tag."""
        table = getattr(attention_inputs, "kv_cache_block_id_device", None)
        if table is None or table.numel() == 0:
            table = getattr(attention_inputs, "kv_cache_block_id", None)
        if table is None or table.numel() == 0 or table.dim() != 2:
            raise RuntimeError(f"PLE cache {tag!r} needs a 2-D physical block table")
        if table.dtype != torch.int32:
            raise RuntimeError(
                f"PLE cache {tag!r} physical block table must be int32, "
                f"got {table.dtype}"
            )
        return table

    @staticmethod
    def _check_physical_blocks(
        tag: str,
        blocks: torch.Tensor,
        pool_rows: int,
    ) -> torch.Tensor:
        if bool((blocks <= 0).any().item()):
            raise RuntimeError(
                f"PLE cache {tag!r} has an unallocated logical page; "
                f"blocks={blocks.tolist()[:32]} pool_rows={pool_rows}"
            )
        if bool((blocks >= pool_rows).any().item()):
            raise RuntimeError(f"PLE cache {tag!r} physical block id exceeds its pool")
        return blocks.to(torch.long)

    @classmethod
    def _physical_blocks(
        cls,
        attention_inputs: PyAttentionInputs,
        tag: str,
        logical_pages: torch.Tensor,
        pool_rows: int,
        device: torch.device,
        *,
        graph_capture: bool = False,
    ) -> torch.Tensor:
        """Resolve one request-local logical page per batch row."""
        table = cls._resolve_block_table(attention_inputs, tag)
        if logical_pages.dim() != 1 or int(table.shape[0]) != int(
            logical_pages.numel()
        ):
            raise RuntimeError(f"PLE cache {tag!r} block-table batch is inconsistent")
        pages = logical_pages.to(device=table.device, dtype=torch.long)
        if graph_capture:
            # The graph runner owns the fixed-size device table and refreshes
            # it each step. Clamp invalid indices to
            # the reserved dummy row so malformed input cannot index outside
            # the pool while capture/replay is in progress.
            if not pages.numel() or not int(table.shape[1]) or pool_rows <= 0:
                raise RuntimeError(f"PLE cache {tag!r} graph table is empty")
            from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_triton import (
                resolve_read_blocks,
            )

            resolved = resolve_read_blocks(table, pages, pool_rows)
            if resolved is not None:
                return resolved.to(device=device)
            valid_page = (pages >= 0) & (pages < int(table.shape[1]))
            safe_pages = pages.clamp(0, int(table.shape[1]) - 1)
            blocks = table.gather(1, safe_pages.unsqueeze(1)).squeeze(1)
            valid_block = (blocks > 0) & (blocks < pool_rows)
            return torch.where(valid_page & valid_block, blocks, 0).to(
                device=device, dtype=torch.long
            )
        if not pages.numel() or bool((pages < 0).any().item()):
            raise RuntimeError(f"PLE cache {tag!r} has an invalid logical page")
        if bool((pages >= int(table.shape[1])).any().item()):
            raise RuntimeError(
                f"PLE cache {tag!r} logical page exceeds its block table"
            )
        return cls._check_physical_blocks(
            tag, table.gather(1, pages.unsqueeze(1)).squeeze(1), pool_rows
        ).to(device=device)

    @classmethod
    def _commit_physical_blocks(
        cls,
        attention_inputs: PyAttentionInputs,
        tag: str,
        logical_pages: torch.Tensor,
        pool_rows: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, list[int]]:
        """Validate commit destinations with one host read, before pool writes."""
        table = cls._resolve_block_table(attention_inputs, tag)
        if logical_pages.dim() != 1 or int(table.shape[0]) != int(
            logical_pages.numel()
        ):
            raise RuntimeError(f"PLE cache {tag!r} block-table batch is inconsistent")
        if not logical_pages.numel():
            raise RuntimeError(f"PLE cache {tag!r} has an invalid logical page")
        if table.shape[1] == 0 or pool_rows <= 0:
            raise RuntimeError(f"PLE cache {tag!r} commit table or pool is empty")
        pages = logical_pages.to(device=table.device, dtype=torch.long)
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_triton import (
            resolve_commit_blocks,
        )

        payload = resolve_commit_blocks(table, pages, pool_rows)
        if payload is not None:
            host = payload.tolist()
            blocks = payload[4:]
        else:
            # Clamp only the temporary lookup. The original page validity remains
            # part of the verdict; an invalid page never becomes a commit plan.
            blocks = table.gather(
                1, pages.clamp(0, int(table.shape[1]) - 1).unsqueeze(1)
            ).squeeze(1)
            verdict = torch.stack(
                [
                    (pages < 0).any(),
                    (pages >= int(table.shape[1])).any(),
                    (blocks <= 0).any(),
                    (blocks >= pool_rows).any(),
                ]
            ).to(torch.long)
            host = torch.cat([verdict, blocks.to(torch.long)]).tolist()
        cls._check_commit_verdict(tag, host[:4], host[4:], pool_rows)
        return blocks.to(device=device, dtype=torch.long), host[4:]

    @staticmethod
    def _check_commit_verdict(tag, flags, host_blocks, pool_rows):
        if flags[0]:
            raise RuntimeError(f"PLE cache {tag!r} has an invalid logical page")
        if flags[1]:
            raise RuntimeError(
                f"PLE cache {tag!r} logical page exceeds its block table"
            )
        if flags[2]:
            raise RuntimeError(
                f"PLE cache {tag!r} has an unallocated logical page; "
                f"blocks={host_blocks[:32]} pool_rows={pool_rows}"
            )
        if flags[3]:
            raise RuntimeError(f"PLE cache {tag!r} physical block id exceeds its pool")

    @classmethod
    def _commit_physical_block_pair(cls, stage, accepted):
        """Fast path only when both tables have valid structural metadata.

        Malformed metadata falls back to the original sequential validation;
        valid metadata retains the state-before-context error precedence.
        """
        from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
            PLE_NGRAM_CTX_TAG,
            PLE_STATE_TAG,
        )
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_triton import (
            resolve_commit_pair,
        )

        tables = []
        for inputs, tag in (
            (stage.state_inputs, PLE_STATE_TAG),
            (stage.ctx_inputs, PLE_NGRAM_CTX_TAG),
        ):
            try:
                tables.append(cls._resolve_block_table(inputs, tag))
            except RuntimeError:
                return None
        if (
            stage.state_pool.device != accepted.device
            or stage.ctx_pool.device != accepted.device
        ):
            return None
        payload = resolve_commit_pair(
            *tables,
            stage.prefixes,
            accepted,
            stage.page_size,
            int(stage.state_pool.shape[0]),
            int(stage.ctx_pool.shape[0]),
        )
        if payload is None:
            return None
        batch = accepted.numel()
        host = payload.tolist()
        state_ids, ctx_ids = host[8 : 8 + batch], host[8 + batch :]
        cls._check_commit_verdict(
            PLE_STATE_TAG, host[:4], state_ids, stage.state_pool.shape[0]
        )
        cls._check_commit_verdict(
            PLE_NGRAM_CTX_TAG, host[4:8], ctx_ids, stage.ctx_pool.shape[0]
        )
        return payload[8 : 8 + batch], state_ids, payload[8 + batch :], ctx_ids

    @classmethod
    def _physical_blocks_for_pairs(
        cls,
        attention_inputs: PyAttentionInputs,
        tag: str,
        rows: torch.Tensor,
        pages: torch.Tensor,
        pool_rows: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve (batch row, logical page) pairs, used for per-page writes.

        Returns the physical blocks plus a mask of the pairs that own a block.
        The allocator keeps the pages it plans to reuse (the request's tail page
        is always among them) and leaves the rest unallocated, so a checkpoint
        may legitimately have no home -- GDN's store_ssm_state_to_block_map
        kernel skips those positions the same way.
        """
        table = cls._resolve_block_table(attention_inputs, tag)
        if (
            rows.dim() != 1
            or pages.dim() != 1
            or int(rows.numel()) != int(pages.numel())
        ):
            raise RuntimeError(f"PLE cache {tag!r} block-table pairs are inconsistent")
        rows_l = rows.to(device=table.device, dtype=torch.long)
        pages_l = pages.to(device=table.device, dtype=torch.long)
        if bool((rows_l < 0).any().item()) or bool(
            (rows_l >= int(table.shape[0])).any().item()
        ):
            raise RuntimeError(f"PLE cache {tag!r} block-table row is out of range")
        if bool((pages_l < 0).any().item()):
            raise RuntimeError(f"PLE cache {tag!r} has an invalid logical page")
        if bool((pages_l >= int(table.shape[1])).any().item()):
            raise RuntimeError(
                f"PLE cache {tag!r} logical page exceeds its block table"
            )
        blocks = table[rows_l, pages_l]
        if bool((blocks >= pool_rows).any().item()):
            raise RuntimeError(f"PLE cache {tag!r} physical block id exceeds its pool")
        return blocks.to(device=device, dtype=torch.long), (blocks > 0).to(
            device=device
        )

    @staticmethod
    def _same_metadata(
        lhs: PyAttentionInputs, rhs: PyAttentionInputs, name: str
    ) -> bool:
        return same_tensor_metadata(getattr(lhs, name, None), getattr(rhs, name, None))
