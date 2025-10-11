import inspect
import itertools
import math
import os
import sys
from unittest import SkipTest, TestCase, main

import flashinfer

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))


def attention_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
            k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
        )
        * sm_scale
    )

    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    lse_ref = torch.logsumexp(logits, -1).transpose(-1, -2)
    p = torch.softmax(logits, dim=-1)
    o_ref = (
        torch.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, kv_len, num_qo_heads, head_dim_vo).float(),
        )
        .contiguous()
        .view(batch_size * qo_len, num_qo_heads, head_dim_vo)
        .to(q)
    )

    return o_ref, lse_ref * math.log2(math.e)


def generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads):
    bs_page_num, page_size, ckv_dim = ckv.shape
    page_num = bs_page_num // batch_size
    _, _, kpe_dim = kpe.shape
    ckv = ckv.view(batch_size, page_num * page_size, ckv_dim)
    kpe = kpe.view(batch_size, page_num * page_size, kpe_dim)
    ckv = ckv[:, :kv_len, :]
    kpe = kpe[:, :kv_len, :]
    k = (
        torch.cat([ckv, kpe], dim=-1)
        .view(-1, 1, ckv_dim + kpe_dim)
        .repeat_interleave(num_heads, dim=1)
    )
    v = ckv.repeat_interleave(num_heads, dim=1)

    return k, v


class MLATest(TestCase):
    BATCH_SIZE = [1]
    # KV_LEN = [0, 17, 33, 96, 97, 114, 514, 1024]
    # QO_LEN = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    KV_LEN = [1024]
    QO_LEN = [128]
    NUM_HEADS = [128]
    # CAUSAL = [False, True]
    CAUSAL = [False]
    # PAGE_SIZE = [1, 16]
    PAGE_SIZE = [1]
    BACKEND = ["auto"]
    USE_CUDA_GRAPH = [False]
    DTYPES = [torch.half]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def batch_prefill_with_ragged_kv_cache_test(
        self,
        batch_size,
        kv_len,
        qo_len,
        num_heads,
        causal,
        backend,
        dtype,
    ):
        device = torch.device("cuda:0")
        torch.manual_seed(42)
        kv_layout = "NHD"
        head_dim_qk = 192
        head_dim_vo = 128
        q = torch.randn(
            batch_size * qo_len, num_heads, head_dim_qk, dtype=dtype, device=device
        )
        q_indptr = (
            torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
        )

        k = torch.zeros(
            batch_size * kv_len, num_heads, head_dim_qk, dtype=dtype, device=device
        )
        v = torch.zeros(
            batch_size * kv_len, num_heads, head_dim_vo, dtype=dtype, device=device
        )
        kv_indptr = (
            torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * kv_len
        )

        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.int8, device=device
        )
        wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, kv_layout, backend=backend
        )
        wrapper.plan(
            q_indptr,
            kv_indptr,
            num_heads,
            num_heads,
            head_dim_qk,
            head_dim_vo=head_dim_vo,
            causal=causal,
        )
        o, lse = wrapper.run_return_lse(q, k, v)

        sm_scale = 1.0 / (head_dim_qk**0.5)
        o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)

        lse_ref = lse_ref.flatten(0, 1)
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)

        # test with pre-allocated output
        o_buffer = torch.empty_like(o)
        lse_buffer = torch.empty_like(lse)
        wrapper.run(q, k, v, out=o_buffer, lse=lse_buffer)
        torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(lse, lse_buffer, rtol=1e-3, atol=1e-3)

    def batch_mla_page_attention_test(
        self,
        batch_size,
        kv_len,
        qo_len,
        num_heads,
        causal,
        page_size,
        backend,
        use_cuda_graph,
        dtype,
    ):
        print("**************************************")
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        for arg in args:
            print(f"{arg} = {values[arg]}")
        print("**************************************")
        device = torch.device("cuda:0")
        torch.manual_seed(42)
        head_dim_ckv = 512
        head_dim_kpe = 64
        q_nope = torch.randn(
            batch_size * qo_len, num_heads, head_dim_ckv, dtype=dtype, device=device
        )
        q_pe = torch.randn(
            batch_size * qo_len, num_heads, head_dim_kpe, dtype=dtype, device=device
        )
        pages_num = math.ceil(kv_len / page_size)
        ckv = torch.randn(
            batch_size * pages_num,
            page_size,
            head_dim_ckv,
            dtype=dtype,
            device=device,
        )
        kpe = torch.randn(
            batch_size * pages_num,
            page_size,
            head_dim_kpe,
            dtype=dtype,
            device=device,
        )
        sm_scale = 1.0 / (
            (128 + 64) ** 0.5
        )  # use head dimension before matrix absorption
        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.int8, device=device
        )
        wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            workspace_buffer,
            backend=backend,
            # use_cuda_graph=True,
            # qo_indptr=torch.empty(batch_size + 1, dtype=torch.int32, device=device),
            # kv_indptr=torch.empty(batch_size + 1, dtype=torch.int32, device=device),
            # kv_indices=torch.empty(1048576, dtype=torch.int32, device=device),
            # kv_len_arr=torch.empty(batch_size, dtype=torch.int32, device=device),
        )

        q_indptr = (
            torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
        )
        kv_indptr = (
            torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
            * pages_num
        )
        kv_indices = torch.arange(
            0, batch_size * pages_num, device=device, dtype=torch.int32
        )
        kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

        if use_cuda_graph:
            kv_indptr_warmup = torch.zeros(
                batch_size + 1, device=device, dtype=torch.int32
            )
            kv_indices_warmup = torch.arange(
                0, batch_size, device=device, dtype=torch.int32
            )
            kv_lens_warmup = torch.full(
                (batch_size,), 0, dtype=torch.int32, device=device
            )
            wrapper.plan(
                q_indptr,
                kv_indptr_warmup,
                kv_indices_warmup,
                kv_lens_warmup,
                num_heads,
                head_dim_ckv,
                head_dim_kpe,
                page_size,
                causal,
                sm_scale,
                q_nope.dtype,
                ckv.dtype,
            )

            # warmup
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)
            torch.cuda.current_stream().wait_stream(s)

            # capture
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)

        wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_lens,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            causal,
            sm_scale,
            q_nope.dtype,
            ckv.dtype,
        )

        if use_cuda_graph:
            o.fill_(0)
            lse.fill_(0)
            g.replay()
        else:
            o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)

        k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)

        q = torch.cat([q_nope, q_pe], dim=-1)
        o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)
        lse_ref = lse_ref.flatten(0, 1)
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
        if kv_len != 0:
            torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)

        # test with pre-allocated output
        o_buffer = torch.empty_like(o)
        lse_buffer = torch.empty_like(lse)
        wrapper.run(q_nope, q_pe, ckv, kpe, out=o_buffer, lse=lse_buffer)
        torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(lse, lse_buffer, rtol=1e-3, atol=1e-3)

    def ref_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """
        Native implementation of scaled dot product attention without mask:
        - query, key, value: [batch_size, seq_len, num_heads, head_size]
        - attn_mask: [batch_size, seq_len, seq_len]
        """
        # query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        attn_weights = scale * torch.matmul(query, key.transpose(2, 3))
        attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
        out = torch.matmul(attn_weights, value).transpose(1, 2)
        return out

    def _run_mla_test(self, dtype: torch.dtype, bs: int = 1, s_q: int = 1):
        torch.set_default_device("cuda:1")
        torch.manual_seed(42)

        # DeepSeek-V2-Lite-Chat config
        self.num_heads = 16  # head_dim = 2048
        self.size_per_head = 128
        self.kv_lora_rank = 512
        self.qk_nope_head_dim = 128
        self.qk_rope_head_dim = 64
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = 128
        self.scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5
        MAX_SEQ_LEN = 1024

        k_nope_weight = torch.randn(
            self.kv_lora_rank, self.num_heads * self.qk_nope_head_dim, dtype=dtype
        )
        torch.nn.init.xavier_uniform_(k_nope_weight)
        v_weight = torch.randn(
            self.kv_lora_rank, self.num_heads * self.v_head_dim, dtype=dtype
        )
        torch.nn.init.xavier_uniform_(v_weight)

        fused_qkv = torch.randn(
            bs,
            s_q,
            self.num_heads * self.q_head_dim
            + self.kv_lora_rank
            + self.qk_rope_head_dim,
            dtype=dtype,
        )
        # kv_offset = self.num_heads * self.size_per_head
        q, compressed_kv = torch.split(
            fused_qkv,
            [
                self.num_heads * self.q_head_dim,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ],
            dim=-1,
        )

        q_view = q.view(bs, s_q, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q_view, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bs, s_q, 1, self.qk_rope_head_dim).transpose(1, 2)
        k_nope = (
            F.linear(compressed_kv, k_nope_weight.transpose(0, 1), None)
            .view(bs, s_q, self.num_heads, self.qk_nope_head_dim)
            .transpose(1, 2)
        )
        value_states = (
            F.linear(compressed_kv, v_weight.transpose(0, 1), None)
            .view(bs, s_q, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
        )
        query_states = k_pe.new_empty(bs, self.num_heads, s_q, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
        key_states = k_pe.new_empty(bs, self.num_heads, s_q, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        ref_out = self.ref_attention(query_states, key_states, value_states, self.scale)
        ref_out = ref_out.transpose(1, 2).view(
            -1, self.num_heads, self.qk_nope_head_dim
        )

        import flashinfer

        batch_size = bs
        page_size = 1
        mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            torch.empty(512 * 1024 * 1024, dtype=torch.int8), backend="auto"
        )
        q_indptr = (
            torch.arange(0, batch_size + 1).to(0).int()
        )  # for decode, each query length is 1
        kv_lens = torch.full((batch_size,), 999, dtype=torch.int32).to(0)
        kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * 999
        kv_indices = torch.arange(0, batch_size * 999).to(0).int()

        mla_wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_lens,
            self.num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            page_size,
            False,  # causal
            self.scale,
            q_nope.dtype,
            compressed_kv.dtype,
        )
        q_nope = q_nope.transpose(1, 2)
        q_pe = q_pe.transpose(1, 2)
        q_nope = q_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        q_pe = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim)

        k_nope_weight = (
            k_nope_weight.view(-1, self.num_heads, self.qk_nope_head_dim)
            .transpose(0, 1)
            .transpose(1, 2)
        )
        q_nope = torch.bmm(q_nope.transpose(0, 1), k_nope_weight)
        q_nope = q_nope.transpose(0, 1)

        compressed_kv = compressed_kv.view(-1, 1, self.kv_lora_rank)
        k_pe = k_pe.transpose(1, 2)
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)

        attn_output = mla_wrapper.run(
            q_nope, q_pe, compressed_kv, k_pe, return_lse=False
        )

        attn_output = attn_output.view(-1, self.num_heads, self.kv_lora_rank)
        v_weight = v_weight.view(-1, self.num_heads, self.qk_nope_head_dim).transpose(
            0, 1
        )
        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), v_weight)
        attn_bmm_output = attn_bmm_output.transpose(0, 1)

    def test_mlp(self):
        self.batch_mla_page_attention_test(
            1, 64, 1, 128, False, 64, "fa2", False, torch.half
        )
        self.batch_prefill_with_ragged_kv_cache_test(
            1, 1, 1, 16, False, "fa2", torch.half
        )
        # for params in itertools.product(self.BATCH_SIZE,
        #                                 self.KV_LEN,
        #                                 self.QO_LEN,
        #                                 self.NUM_HEADS,
        #                                 self.CAUSAL,
        #                                 self.PAGE_SIZE,
        #                                 self.BACKEND,
        #                                 self.USE_CUDA_GRAPH,
        #                                 self.DTYPES):
        #     with self.subTest(batch_size=params[0],
        #                       kv_len=params[1],
        #                       qo_len=params[2],
        #                       num_heads=params[3],
        #                       causal=params[4],
        #                       page_size=params[5],
        #                       backend=params[6],
        #                       use_cuda_graph=params[7],
        #                       dtype=params[8]):
        #         self.batch_mla_page_attention_test(*params)


if __name__ == "__main__":
    main()
