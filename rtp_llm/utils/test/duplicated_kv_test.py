import math
import random
import unittest

import torch

from rtp_llm.utils.model_weight import get_sp_tensor, sp_head_qk_norm


# This test validates the Q-to-KV head mapping after tensor splitting.
#
# For each test configuration:
# 1. Random query/key/value tensors are created once for the given
#    (head_dim, q_head, kv_head) setting.
# 2. For each tp/tp_rank, the tensors are split in the same head-wise way
#    expected by the test:
#       - Q is split across tp ranks
#       - K/V are split across gcd(kv_head, tp) ranks
# 3. One local query head is randomly selected from the current tp rank.
# 4. Its corresponding KV head is computed by grouped-query attention mapping:
#       kv_idx = q_idx // (q_head // kv_head)
# 5. The outputs of get_sp_tensor and sp_head_qk_norm are checked to ensure:
#       - the selected Q head is placed correctly after splitting
#       - the mapped K head is placed correctly after splitting
#       - the mapped V head is placed correctly after splitting
#
# The test focuses on whether the head correspondence remains correct after
# applying the splitting logic, rather than comparing the whole output tensor.
class TestDuplicatedKV(unittest.TestCase):
    def test_random_qhead_kv_correspondence_after_split(self):
        head_dims = [2, 31, 32, 128]
        head_configs = [(32, 2), (32, 4), (8, 4), (8, 2)]
        tps = [1, 2, 4, 8]

        for head_dim in head_dims:
            for q_head, kv_head in head_configs:
                self.assertEqual(q_head % kv_head, 0)
                group_size = q_head // kv_head

                query = torch.randn(q_head, head_dim)
                key = torch.randn(kv_head, head_dim)
                value = torch.randn(kv_head, head_dim)

                qk_input = torch.cat([query.reshape(-1), key.reshape(-1)], dim=0)
                qkv_input = torch.cat(
                    [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0
                )

                for tp in tps:
                    for tp_rank in range(tp):
                        with self.subTest(
                            head_dim=head_dim,
                            q_head=q_head,
                            kv_head=kv_head,
                            tp=tp,
                            tp_rank=tp_rank,
                        ):
                            qk_out = sp_head_qk_norm(
                                qk_input,
                                tp=tp,
                                tp_rank=tp_rank,
                                head_num=q_head,
                                head_num_kv=kv_head,
                                size_per_head=head_dim,
                            )

                            qkv_out = get_sp_tensor(
                                qkv_input,
                                head_num=q_head,
                                head_num_kv=kv_head,
                                size_per_head=head_dim,
                                tp=tp,
                                tp_rank=tp_rank,
                            )

                            kv_tp = math.gcd(kv_head, tp)
                            kv_rank = tp_rank // (tp // kv_tp)

                            q_chunks = torch.chunk(query, tp, dim=0)
                            local_query = q_chunks[tp_rank]

                            k_chunks = torch.chunk(key, kv_tp, dim=0)
                            v_chunks = torch.chunk(value, kv_tp, dim=0)
                            local_key = k_chunks[kv_rank]
                            local_value = v_chunks[kv_rank]

                            q_len = local_query.numel()
                            k_len = local_key.numel()
                            v_len = local_value.numel()

                            q_part_from_qk = qk_out[:, :q_len].reshape(-1, head_dim)
                            k_part_from_qk = qk_out[:, q_len:q_len + k_len].reshape(-1, head_dim)

                            q_part_from_qkv = qkv_out[:, :q_len].reshape(-1, head_dim)
                            k_part_from_qkv = qkv_out[:, q_len:q_len + k_len].reshape(-1, head_dim)
                            v_part_from_qkv = qkv_out[:, q_len + k_len:q_len + k_len + v_len].reshape(-1, head_dim)

                            if local_query.shape[0] == 0:
                                continue

                            local_q_idx = random.randrange(local_query.shape[0])
                            global_q_idx = sum(chunk.shape[0] for chunk in q_chunks[:tp_rank]) + local_q_idx
                            expected_kv_idx = global_q_idx // group_size

                            self.assertTrue(0 <= expected_kv_idx < kv_head)

                            self.assertTrue(
                                torch.allclose(q_part_from_qk[local_q_idx], query[global_q_idx]),
                                msg=(
                                    f"Q head mismatch in sp_head_qk_norm: "
                                    f"head_dim={head_dim}, q_head={q_head}, kv_head={kv_head}, "
                                    f"tp={tp}, tp_rank={tp_rank}, global_q_idx={global_q_idx}"
                                ),
                            )
                            self.assertTrue(
                                torch.allclose(q_part_from_qkv[local_q_idx], query[global_q_idx]),
                                msg=(
                                    f"Q head mismatch in get_sp_tensor: "
                                    f"head_dim={head_dim}, q_head={q_head}, kv_head={kv_head}, "
                                    f"tp={tp}, tp_rank={tp_rank}, global_q_idx={global_q_idx}"
                                ),
                            )

                            kv_prefix = sum(chunk.shape[0] for chunk in k_chunks[:kv_rank])
                            if kv_prefix <= expected_kv_idx < kv_prefix + local_key.shape[0]:
                                local_kv_idx = expected_kv_idx - kv_prefix

                                self.assertTrue(
                                    torch.allclose(
                                        k_part_from_qk[local_kv_idx], key[expected_kv_idx]
                                    ),
                                    msg=(
                                        f"K head mismatch in sp_head_qk_norm: "
                                        f"head_dim={head_dim}, q_head={q_head}, kv_head={kv_head}, "
                                        f"tp={tp}, tp_rank={tp_rank}, kv_rank={kv_rank}, "
                                        f"global_q_idx={global_q_idx}, expected_kv_idx={expected_kv_idx}"
                                    ),
                                )
                                self.assertTrue(
                                    torch.allclose(
                                        k_part_from_qkv[local_kv_idx], key[expected_kv_idx]
                                    ),
                                    msg=(
                                        f"K head mismatch in get_sp_tensor: "
                                        f"head_dim={head_dim}, q_head={q_head}, kv_head={kv_head}, "
                                        f"tp={tp}, tp_rank={tp_rank}, kv_rank={kv_rank}, "
                                        f"global_q_idx={global_q_idx}, expected_kv_idx={expected_kv_idx}"
                                    ),
                                )
                                self.assertTrue(
                                    torch.allclose(
                                        v_part_from_qkv[local_kv_idx], value[expected_kv_idx]
                                    ),
                                    msg=(
                                        f"V head mismatch in get_sp_tensor: "
                                        f"head_dim={head_dim}, q_head={q_head}, kv_head={kv_head}, "
                                        f"tp={tp}, tp_rank={tp_rank}, kv_rank={kv_rank}, "
                                        f"global_q_idx={global_q_idx}, expected_kv_idx={expected_kv_idx}"
                                    ),
                                )