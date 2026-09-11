"""EP subscriptions must not change the stacked loader's collective sequence."""

import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from fastsafetensors.frameworks._torch import TorchProcessGroup
from safetensors.torch import save_file

from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.per_expert_parallel_loader import PerExpertParallelLoader
from rtp_llm.model_loader.tensor_source import TensorCollector
from rtp_llm.utils.model_weight import CkptWeightInfo, W

STACKED_KEY = "model.layers.0.mlp.experts.gate_up_proj"
EXPERTS = 4


def _rank_config(rank, ep_size=2):
    load_config = LoadConfig.model_construct(
        ep_rank=rank, ep_size=ep_size, phy2log=None
    )
    weights = [
        MoeAtomicWeight(
            name=name,
            weights=[CkptWeightInfo(STACKED_KEY)],
            config=MoeConfig(expert_num=EXPERTS),
            stacked_ckpt_keys=True,
        )
        for name in (W.moe_w1, W.moe_s1)
    ]
    selected = list(load_config.get_selected_experts(0, EXPERTS))
    subscribed_keys = weights[0].get_tensor_names(0, load_config)
    # Model online PTPC: the scale descriptor shares the physical key, but
    # the collector only requests the kernel keys; scales are derived later.
    collector = TensorCollector(subscribed_keys, None)
    weight_infos = [ModelLoader.WeightInfo(w, 0, collector) for w in weights]
    return (
        ModelLoader._build_stacked_key_config(weight_infos),
        subscribed_keys,
        selected,
    )


def _nccl_load_worker(rank, checkpoint, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        config, subscribed_keys, selected = _rank_config(rank)
        peer_configs = [None, None]
        dist.all_gather_object(peer_configs, config)
        assert peer_configs[0] == peer_configs[1]
        calls = []
        broadcast = TorchProcessGroup.broadcast

        def recorded_broadcast(group, tensor, src_rank):
            calls.append((src_rank, tuple(tensor.get_raw().shape)))
            return broadcast(group, tensor, src_rank)

        loader = PerExpertParallelLoader(
            config,
            subscribed_keys=subscribed_keys,
            pg=dist.group.WORLD,
            hf_weights_files=[checkpoint],
            device=f"cuda:{rank}",
            nogds=True,
            set_numa=False,
            use_tqdm_on_load=False,
        )
        try:
            with patch.object(TorchProcessGroup, "broadcast", recorded_broadcast):
                outputs = dict(loader.iterate_weights())
        finally:
            loader.loader.close()
        assert set(outputs) == subscribed_keys
        template = config[STACKED_KEY][0]
        expected = torch.arange(EXPERTS * 6, dtype=torch.float32).reshape(EXPERTS, 2, 3)
        for eid in selected:
            torch.testing.assert_close(
                outputs[template.format(expert_id=eid)].cpu(), expected[eid]
            )
        peer_calls = [None, None]
        dist.all_gather_object(peer_calls, calls)
        assert peer_calls[0] == peer_calls[1]
        assert len(calls) == EXPERTS, calls
        assert all(shape == (2, 3) for _, shape in calls), calls
    finally:
        dist.destroy_process_group()


class TestPerExpertParallelLoader(unittest.TestCase):
    def test_rank_configs_keep_all_templates(self):
        config0, subscribed0, selected0 = _rank_config(0)
        config1, subscribed1, selected1 = _rank_config(1)
        self.assertEqual(selected0, [0, 1])
        self.assertEqual(selected1, [2, 3])
        self.assertTrue(subscribed0.isdisjoint(subscribed1))
        self.assertEqual(config0, config1)
        self.assertEqual(len(config0[STACKED_KEY]), 2)

    def test_model_loader_passes_global_config_and_local_subscriptions(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                config, subscribed_keys, _ = _rank_config(rank)
                weight = MoeAtomicWeight(
                    name=W.moe_w1,
                    weights=[CkptWeightInfo(STACKED_KEY)],
                    config=MoeConfig(expert_num=EXPERTS),
                    stacked_ckpt_keys=True,
                )
                wi = ModelLoader.WeightInfo(weight, 0, None)
                loader = ModelLoader.__new__(ModelLoader)
                loader._create_model_weights = MagicMock()
                loader._generate_weight_info = MagicMock(
                    return_value=({key: wi for key in subscribed_keys}, [wi])
                )
                loader._is_online_ptpc = MagicMock(return_value=False)
                iterator = MagicMock(side_effect=RuntimeError("captured config"))
                loader._load_config = SimpleNamespace(
                    database=SimpleNamespace(fastsafetensors_weights_iterator=iterator)
                )
                with self.assertRaisesRegex(RuntimeError, "captured config"):
                    loader._load_from_fastsafetensor("cuda")
                iterator.assert_called_once_with(
                    "cuda",
                    True,
                    stacked_key_config={STACKED_KEY: config[STACKED_KEY][:1]},
                    subscribed_keys=subscribed_keys,
                )

    def test_only_subscribed_outputs_are_cloned(self):
        for size, rank in ((1, 0), (2, 0), (2, 1)):
            for selection in ("local", "empty", "unfiltered"):
                with self.subTest(size=size, rank=rank, selection=selection):
                    config, subscribed, _ = _rank_config(rank, size)
                    if selection == "empty":
                        subscribed = set()
                    elif selection == "unfiltered":
                        subscribed = None
                    loader = PerExpertParallelLoader.__new__(PerExpertParallelLoader)
                    loader.stacked_key_config = config
                    loader.subscribed_keys = subscribed
                    tensor = MagicMock()
                    factory = SimpleNamespace(
                        metadata=SimpleNamespace(
                            tensors={
                                STACKED_KEY: SimpleNamespace(
                                    shape=(EXPERTS, 2, 3), dtype=torch.float32
                                )
                            }
                        ),
                        tensors={STACKED_KEY: tensor},
                        framework=MagicMock(),
                        device="cpu",
                        free_dev_ptrs=MagicMock(),
                    )
                    pg = MagicMock()
                    pg.size.return_value = size
                    pg.rank.return_value = rank
                    fb = SimpleNamespace(
                        _get_rank_lidx=lambda key: (0, 0),
                        rank_loaders={0: {0: factory}},
                        pg=pg,
                        auto_mem_delete=True,
                        instantiated={0: {0: {}}},
                    )
                    result = dict(
                        loader._broadcast_per_expert(
                            SimpleNamespace(fb=fb), STACKED_KEY
                        )
                    )
                    expected_keys = (
                        subscribed
                        if subscribed is not None
                        else {
                            t.format(expert_id=e)
                            for t in config[STACKED_KEY]
                            for e in range(EXPERTS)
                        }
                    )
                    self.assertEqual(set(result), expected_keys)
                    self.assertEqual(
                        pg.broadcast.call_count, EXPERTS if size > 1 else 0
                    )
                    if size == 1:
                        output_tensor = tensor.__getitem__.return_value
                    elif rank == 0:
                        source_slice = tensor.__getitem__.return_value
                        output_tensor = (
                            source_slice.clone.return_value.detach.return_value
                        )
                    else:
                        output_tensor = factory.framework.get_empty_tensor.return_value
                    self.assertEqual(output_tensor.clone.call_count, len(expected_keys))
                    factory.free_dev_ptrs.assert_called_once_with()

    def test_two_rank_stacked_checkpoint_nccl(self):
        self.assertGreaterEqual(
            torch.cuda.device_count(),
            2,
            "test target must be allocated two CUDA GPUs",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "model.safetensors")
            save_file(
                {
                    STACKED_KEY: torch.arange(EXPERTS * 6, dtype=torch.float32).reshape(
                        EXPERTS, 2, 3
                    )
                },
                checkpoint,
            )
            mp.spawn(
                _nccl_load_worker,
                args=(checkpoint, "file://" + os.path.join(directory, "rendezvous")),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
