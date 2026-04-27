"""Vendored Triton K-Means (Euclidean only) for PQ attention.

Inlined from flash-kmeans (https://github.com/.../flash_kmeans), Euclidean path:
- batch_kmeans_Euclid          (top-level entry)
- _euclid_iter                 (single iteration: assign + update + shift)
- euclid_assign_triton         (Triton nearest-centroid assignment)
- triton_centroid_update_sorted_euclid  (Triton sorted-chunk centroid update)

Cosine / Dot variants and large-N streaming are intentionally omitted.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# ============================================================================
# Assign kernel: nearest-centroid IDs in Euclidean distance
# ============================================================================

_TUNE_CONFIGS = [
    triton.Config({"BLOCK_N": BN, "BLOCK_K": BK}, num_stages=num_stages, num_warps=wp)
    for BN in [32, 64, 128]
    for BK in [32, 64, 128]
    for wp in [4, 8]
    for num_stages in [1, 2, 4]
]


def _cfg_keep(conf):
    BN = conf.kwargs["BLOCK_N"]
    BK = conf.kwargs["BLOCK_K"]
    if BN * BK < 32 * 32 and conf.num_warps > 4:
        return False
    return True


_TUNE_CONFIGS = list(filter(_cfg_keep, _TUNE_CONFIGS))


def _heuristic_euclid_config(
    N: int,
    K: int,
    D: int,
    *,
    device: Optional[torch.device] = None,
):
    """Architecture-aware heuristic config (skip autotune)."""
    if device is None:
        device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_properties(device).name.upper()

    if "H200" in gpu_name:
        block_n = 128
        block_k = 64
        num_warps = 4
        num_stages = 1

        if D >= 512:
            block_n = 128
            block_k = 64
            num_warps = 8
            num_stages = 1
        elif D >= 256:
            block_n = 128
            block_k = 64
            num_warps = 4
            num_stages = 2
        else:
            if K >= 4096:
                block_k = 128
                if D >= 128:
                    num_warps = 8
                    num_stages = 2
                else:
                    num_warps = 4
                    num_stages = 4
            else:
                block_k = 64
                num_warps = 4
                num_stages = 1

        if D <= 64 and K >= 4096:
            block_n = 64
            block_k = 128
            num_warps = 4
            num_stages = 4

        if N < 65536:
            block_n = 64

        return {
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }

    if "H100" in gpu_name:
        block_n = 128
        block_k = 64
        num_warps = 4
        num_stages = 1

        if D >= 512:
            block_n = 128
            block_k = 64
            num_warps = 8
            num_stages = 1
        elif D >= 256:
            block_n = 128
            block_k = 64
            if K <= 1024:
                num_warps = 8
                num_stages = 1
            elif K <= 16384:
                num_warps = 4
                num_stages = 1
            else:
                num_warps = 8
                num_stages = 1
        else:
            if D <= 64:
                if K <= 1024:
                    block_k = 64
                    num_warps = 4
                    num_stages = 2
                elif K <= 16384:
                    block_k = 64
                    num_warps = 4
                    num_stages = 2
                elif K <= 65536:
                    block_k = 128
                    num_warps = 4
                    num_stages = 4
                else:
                    block_k = 64
                    num_warps = 4
                    num_stages = 4
            else:
                if K <= 1024:
                    block_k = 64
                    num_warps = 4
                    num_stages = 1
                elif K <= 65536:
                    block_k = 128
                    num_warps = 8
                    num_stages = 2
                else:
                    block_k = 64
                    num_warps = 4
                    num_stages = 4

        if N < 65536:
            block_n = 64

        return {
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }

    if "A100" in gpu_name:
        block_n = 128
        block_k = 32
        num_warps = 4
        num_stages = 2

        if D == 128:
            if N <= 65536:
                block_k = 64
        elif D == 256:
            if K >= 65536:
                block_k = 32
                num_stages = 4
            elif K >= 1024 and N <= 262144:
                block_k = 64
                num_stages = 4

        return {
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }

    return {
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "num_warps": 4,
        "num_stages": 1,
    }


@triton.jit
def _euclid_assign_kernel(
    x_ptr,
    c_ptr,
    x_sq_ptr,
    c_sq_ptr,
    out_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_csq_b: tl.constexpr,
    stride_csq_k: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_b = pid_b.to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_offsets = n_offsets.to(tl.int64)
    n_mask = n_offsets < N

    offs_d = tl.arange(0, D).to(tl.int64)
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)

    xsq_ptrs = x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n
    x_sq_tile = tl.load(xsq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_offsets = k_offsets.to(tl.int64)
        k_mask = k_offsets < K

        c_ptrs = (
            c_ptr
            + pid_b * stride_c_b
            + k_offsets[None, :] * stride_c_k
            + offs_d[:, None] * stride_c_d
        )
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)

        csq_ptrs = c_sq_ptr + pid_b * stride_csq_b + k_offsets * stride_csq_k
        cent_sq = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        cross = tl.dot(x_tile, c_tile).to(tl.float32)
        dist = x_sq_tile[:, None] + cent_sq[None, :] - 2.0 * cross
        dist = tl.maximum(dist, 0.0)
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


_euclid_assign_kernel_autotuned = triton.autotune(_TUNE_CONFIGS, key=["N", "K"])(
    _euclid_assign_kernel
)


def euclid_assign_triton(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
    out: torch.Tensor = None,
    c_sq: torch.Tensor = None,
    *,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    config: Optional[dict] = None,
    use_heuristic: bool = True,
) -> torch.Tensor:
    """Nearest-centroid indices via Triton kernel.

    x         : (B, N, D) fp16/fp32 (CUDA)
    centroids : (B, K, D) same dtype/device as x
    x_sq      : (B, N)    fp32 — ||x||^2 per point
    """
    assert x.is_cuda and centroids.is_cuda and x_sq.is_cuda
    assert centroids.dtype == x.dtype, "centroids dtype mismatch"

    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D), "centroids shape mismatch"
    assert x_sq.shape == (B, N), "x_sq shape mismatch"

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)
    if c_sq is None:
        c_sq = (centroids.to(torch.float32) ** 2).sum(-1)

    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_csq_b, stride_csq_k = c_sq.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    selected_config = None
    if config is not None:
        selected_config = config
    elif num_warps is not None or num_stages is not None:
        if num_warps is None or num_stages is None:
            raise ValueError("num_warps and num_stages must be set together")
        selected_config = {
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    elif use_heuristic:
        selected_config = _heuristic_euclid_config(N, K, D, device=x.device)

    if selected_config is not None:
        _euclid_assign_kernel[grid](
            x,
            centroids,
            x_sq,
            c_sq,
            out,
            B,
            N,
            K,
            D,
            stride_x_b,
            stride_x_n,
            stride_x_d,
            stride_c_b,
            stride_c_k,
            stride_c_d,
            stride_xsq_b,
            stride_xsq_n,
            stride_csq_b,
            stride_csq_k,
            stride_out_b,
            stride_out_n,
            BLOCK_N=selected_config["BLOCK_N"],
            BLOCK_K=selected_config["BLOCK_K"],
            num_warps=selected_config["num_warps"],
            num_stages=selected_config["num_stages"],
        )
    else:
        _euclid_assign_kernel_autotuned[grid](
            x,
            centroids,
            x_sq,
            c_sq,
            out,
            B,
            N,
            K,
            D,
            stride_x_b,
            stride_x_n,
            stride_x_d,
            stride_c_b,
            stride_c_k,
            stride_c_d,
            stride_xsq_b,
            stride_xsq_n,
            stride_csq_b,
            stride_csq_k,
            stride_out_b,
            stride_out_n,
        )
    return out


# ============================================================================
# Centroid update: sorted-chunk Triton kernel (Euclidean)
# ============================================================================


@triton.jit
def _centroid_update_chunk_kernel(
    x_ptr,
    sorted_idx_ptr,
    sorted_cluster_ptr,
    sum_ptr,
    count_ptr,
    stride_x_b,
    stride_x_n,
    stride_x_d,
    stride_idx_b,
    stride_idx_n,
    stride_cluster_b,
    stride_cluster_n,
    stride_sum_b,
    stride_sum_k,
    stride_sum_d,
    stride_count_b,
    stride_count_k,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Each program: BLOCK_N consecutive sorted tokens, one atomic add per cluster run."""
    pid_chunk = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    b = pid_b.to(tl.int64)
    chunk_start = (pid_chunk * BLOCK_N).to(tl.int64)

    if chunk_start >= N:
        return

    idx_batch_base = sorted_idx_ptr + b * stride_idx_b
    cid_batch_base = sorted_cluster_ptr + b * stride_cluster_b
    x_batch_base = x_ptr + b * stride_x_b

    offs_token = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_dim = tl.arange(0, D).to(tl.int64)

    token_idx = chunk_start + offs_token
    valid_tok = token_idx < N
    first_token_idx = chunk_start
    last_token_idx = tl.minimum(chunk_start + BLOCK_N, N) - 1

    first_id = tl.load(cid_batch_base + first_token_idx)
    last_id = tl.load(cid_batch_base + last_token_idx)
    all_ids = tl.load(
        cid_batch_base + token_idx * stride_cluster_n, mask=valid_tok, other=-1
    )

    all_tokens_idxs = tl.load(
        idx_batch_base + token_idx * stride_idx_n, mask=valid_tok, other=-1
    )
    all_tokens_idxs = all_tokens_idxs.to(tl.int64)

    for cid in range(first_id, last_id + 1):
        cluster_mask = all_ids == cid
        cluster_size = tl.sum(cluster_mask.to(tl.int32))
        if cluster_size != 0:
            row_ptrs = (
                x_batch_base
                + all_tokens_idxs[:, None] * stride_x_n
                + offs_dim[None, :] * stride_x_d
            )
            cluster_feats = tl.load(row_ptrs, mask=cluster_mask[:, None], other=0.0)
            cluster_feats = cluster_feats.to(tl.float32)
            sum_feats = tl.sum(cluster_feats, axis=0)
            dest_ptr = (
                sum_ptr
                + b * stride_sum_b
                + cid * stride_sum_k
                + offs_dim * stride_sum_d
            )
            tl.atomic_add(dest_ptr, sum_feats)
            tl.atomic_add(
                count_ptr + b * stride_count_b + cid * stride_count_k, cluster_size
            )


def triton_centroid_update_sorted_euclid(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    old_centroids: torch.Tensor,
    *,
    BLOCK_N: int = 256,
    centroid_sums: torch.Tensor = None,
    centroid_cnts: torch.Tensor = None,
    calculate_new: bool = True,
):
    """Fast Euclidean centroid update; sorts cluster IDs internally."""
    assert x.is_cuda and cluster_ids.is_cuda
    B, N, D = x.shape
    K = old_centroids.shape[1]

    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
    sorted_idx_int = sorted_idx.to(torch.int32)

    if centroid_sums is None:
        centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    else:
        assert centroid_sums.shape == (B, K, D)

    if centroid_cnts is None:
        centroid_cnts = torch.zeros((B, K), device=x.device, dtype=torch.int32)
    else:
        assert centroid_cnts.shape == (B, K)

    grid = (triton.cdiv(N, BLOCK_N), B)
    _centroid_update_chunk_kernel[grid](
        x,
        sorted_idx_int,
        sorted_cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_cnts,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        sorted_idx_int.stride(0),
        sorted_idx_int.stride(1),
        sorted_cluster_ids.stride(0),
        sorted_cluster_ids.stride(1),
        centroid_sums.stride(0),
        centroid_sums.stride(1),
        centroid_sums.stride(2),
        centroid_cnts.stride(0),
        centroid_cnts.stride(1),
        B,
        N,
        D,
        K,
        BLOCK_N=BLOCK_N,
    )

    if calculate_new:
        counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
        centroids = centroid_sums / counts_f
        empty_mask = (centroid_cnts == 0).unsqueeze(-1)
        centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
        return centroids.to(x.dtype)
    else:
        return None


# ============================================================================
# Top-level: batched Euclidean K-Means
# ============================================================================


def _euclid_iter(x, x_sq, centroids, use_heuristic=True):
    cluster_ids = euclid_assign_triton(x, centroids, x_sq, use_heuristic=use_heuristic)
    centroids_new = triton_centroid_update_sorted_euclid(x, cluster_ids, centroids)
    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids


def batch_kmeans_Euclid(
    x,
    n_clusters,
    max_iters=100,
    tol=0.0,
    init_centroids=None,
    verbose=False,
    *,
    use_heuristic=True,
):
    """Batched Euclidean K-Means.

    x: (B, N, D)
    Returns:
        cluster_ids: (B, N) int32
        centroids: (B, n_clusters, D)
        n_iters: int
    """
    B, N, D = x.shape

    x_sq = (x**2).sum(dim=-1)

    if init_centroids is None:
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(x, dim=1, index=indices[..., None].expand(-1, -1, D))
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)

    for it in range(max_iters):
        centroids_new, center_shift, cluster_ids = _euclid_iter(
            x, x_sq, centroids, use_heuristic
        )
        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new.clone()

    return cluster_ids, centroids, it + 1
