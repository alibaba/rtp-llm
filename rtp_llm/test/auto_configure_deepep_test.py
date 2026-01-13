from unittest import TestCase, main

from rtp_llm.config.py_config_modules import DeepEPConfig
from rtp_llm.config.server_config_setup import auto_configure_deepep
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.ops import MoeConfig, RoleType
from pytest import mark


@mark.gpu
class AutoConfigureDeepepTest(TestCase):
    """Test cases for auto_configure_deepep function"""

    def setUp(self):
        """Set up test environment before each test"""
        # Create fresh config objects for each test
        self.moe_config = MoeConfig()
        self.deep_ep_config = DeepEPConfig()

    def _setup_parallel_info(
        self, world_size: int, tp_size: int, ep_size: int, local_world_size: int = 8
    ):
        """Helper to set up parallel info

        Ensures world_size % tp_size == 0 to match real-world constraints.
        """
        assert (
            world_size % tp_size == 0
        ), f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
        g_parallel_info.world_size = world_size
        g_parallel_info.tp_size = tp_size
        g_parallel_info.ep_size = ep_size
        g_parallel_info.local_world_size = local_world_size

    def _assert_deepep_config(self, moe: bool, low_latency: bool, internode: bool):
        """Helper to assert DeepEP configuration values in moe_config"""
        self.assertEqual(self.moe_config.use_deepep_moe, moe)
        self.assertEqual(self.moe_config.use_deepep_low_latency, low_latency)
        self.assertEqual(self.moe_config.use_deepep_internode, internode)

    def test_use_all_gather_disables_all_deepep(self):
        """Test: USE_ALL_GATHER enabled should disable all DeepEP settings"""
        # Setup: ep_size == tp_size and USE_ALL_GATHER is True
        self._setup_parallel_info(
            world_size=8, tp_size=8, ep_size=8, local_world_size=8
        )
        self.moe_config.use_all_gather = True
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        # Verify: All DeepEP settings should be 0
        self._assert_deepep_config(moe=False, low_latency=False, internode=False)
        # test set use_all_gather = False cause use_deepep_moe = True
        self.moe_config.use_all_gather = False
        self.moe_config.use_deepep_moe = False
        self.moe_config.use_deepep_low_latency = False
        self.moe_config.use_deepep_internode = False

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        # Verify: use_deepep_moe should be True
        self._assert_deepep_config(moe=True, low_latency=False, internode=False)

    def test_inference_single_gpu(self):
        """Test: Non-PD separation + Inference node + Single GPU (1TP): 0, 0, 0"""
        self._setup_parallel_info(
            world_size=1, tp_size=1, ep_size=1, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        self._assert_deepep_config(moe=False, low_latency=False, internode=False)

    def test_inference_multi_node_multi_gpu(self):
        """Test: Non-PD separation + Inference node + Multi-node multi-GPU: 1, 0, 1"""
        # world_size=16, local_world_size=8 => num_nodes = (16 + 8 - 1) // 8 = 2
        self._setup_parallel_info(
            world_size=16, tp_size=8, ep_size=16, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        self._assert_deepep_config(moe=True, low_latency=False, internode=True)
        # test set use_all_gather = True not influence result (ep_size != tp_size)
        self.moe_config.use_all_gather = True
        self.moe_config.use_deepep_moe = False
        self.moe_config.use_deepep_low_latency = False
        self.moe_config.use_deepep_internode = False

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        self._assert_deepep_config(moe=True, low_latency=False, internode=True)

    def test_prefill_single_gpu(self):
        """Test: PD separation + Prefill node + Single-node single-GPU (1TP): 0, 0, 0"""
        self._setup_parallel_info(
            world_size=1, tp_size=1, ep_size=1, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.PREFILL

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        self._assert_deepep_config(moe=False, low_latency=False, internode=False)
        # test set use_all_gather = False (single GPU, should still be False)
        self.moe_config.use_all_gather = False
        self.moe_config.use_deepep_moe = False
        self.moe_config.use_deepep_low_latency = False
        self.moe_config.use_deepep_internode = False

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )
        self._assert_deepep_config(moe=False, low_latency=False, internode=False)

    def test_decode_single_gpu(self):
        """Test: PD separation + Decode node + Single-node single-GPU (1TP): 0, 0, 0"""
        self._setup_parallel_info(
            world_size=1, tp_size=1, ep_size=1, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.DECODE

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        self._assert_deepep_config(moe=False, low_latency=False, internode=False)

    def test_prefill_single_node_multi_gpu(self):
        """Test: PD separation + Prefill node + Single-node multi-GPU (2-8 GPUs): 1, 0, 0"""
        self._setup_parallel_info(
            world_size=8, tp_size=4, ep_size=8, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.PREFILL

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        self._assert_deepep_config(moe=True, low_latency=False, internode=False)

    def test_decode_single_node_multi_gpu(self):
        """Test: PD separation + Decode node + Single-node multi-GPU (2-8 GPUs): 1, 1, 0"""
        self._setup_parallel_info(
            world_size=8, tp_size=4, ep_size=8, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.DECODE

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        self._assert_deepep_config(moe=True, low_latency=True, internode=False)

    def test_prefill_multi_node_multi_gpu(self):
        """Test: PD separation + Prefill node + Multi-node multi-GPU: 1, 0, 1"""
        # world_size=16, local_world_size=8 => num_nodes = (16 + 8 - 1) // 8 = 2
        self._setup_parallel_info(
            world_size=16, tp_size=8, ep_size=16, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.PREFILL

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        self._assert_deepep_config(moe=True, low_latency=False, internode=True)

    def test_decode_multi_node_multi_gpu(self):
        """Test: PD separation + Decode node + Multi-node multi-GPU: 1, 1, 1"""
        # world_size=16, local_world_size=8 => num_nodes = (16 + 8 - 1) // 8 = 2
        self._setup_parallel_info(
            world_size=16, tp_size=8, ep_size=16, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.DECODE

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        self._assert_deepep_config(moe=True, low_latency=True, internode=True)

    def test_world_size_exceeds_local_world_size(self):
        """Test: world_size > local_world_size triggers multi-node logic"""
        # world_size=16, local_world_size=8 => world_size > local_world_size => is_multi_node=True
        self._setup_parallel_info(
            world_size=16, tp_size=8, ep_size=16, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.DECODE

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        # Should treat as multi-node due to world_size > local_world_size
        self._assert_deepep_config(moe=True, low_latency=True, internode=True)

    def test_8_gpu_single_node(self):
        """Test: 8 GPU single node should not be treated as multi-node"""
        # world_size=8, local_world_size=8 => num_nodes = (8 + 8 - 1) // 8 = 1
        # world_size <= local_world_size => is_multi_node=False
        self._setup_parallel_info(
            world_size=8, tp_size=8, ep_size=8, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.DECODE

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        # Should treat as single-node multi-GPU
        self._assert_deepep_config(moe=True, low_latency=True, internode=False)

    def test_16_gpu_single_node_with_16_local_devices(self):
        """Test: 16 GPU single node (world_size=16, local_world_size=16) should NOT be multi-node"""
        # world_size=16, local_world_size=16 => num_nodes = (16 + 16 - 1) // 16 = 1
        # world_size <= local_world_size => is_multi_node=False
        self._setup_parallel_info(
            world_size=16, tp_size=16, ep_size=16, local_world_size=16
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.DECODE

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        # Should treat as single-node multi-GPU (not multi-node)
        # Because: num_nodes=1 and world_size == local_world_size
        self._assert_deepep_config(moe=True, low_latency=True, internode=False)

    def test_use_all_gather_false_but_ep_not_equal_tp(self):
        """Test: USE_ALL_GATHER true but ep_size != tp_size, should not disable DeepEP"""
        self._setup_parallel_info(
            world_size=8, tp_size=4, ep_size=8, local_world_size=8
        )
        self.moe_config.use_all_gather = True  # Config is True
        role_type = RoleType.DECODE

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        # Since ep_size != tp_size, use_all_gather should not be enabled
        # Should apply normal rules for DECODE + single-node multi-GPU
        self._assert_deepep_config(moe=True, low_latency=True, internode=False)

    def test_16_gpu_with_8_local_devices_is_multi_node(self):
        """Test: 16 GPUs with only 8 local devices should be treated as multi-node"""
        # world_size=16, local_world_size=8
        # num_nodes = (16 + 8 - 1) // 8 = 2
        # world_size > local_world_size => is_multi_node=True
        self._setup_parallel_info(
            world_size=16, tp_size=8, ep_size=16, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.DECODE

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        # Should treat as multi-node
        # Because: num_nodes=2 or world_size(16) > local_world_size(8)
        self._assert_deepep_config(moe=True, low_latency=True, internode=True)

    def test_32_gpu_with_16_local_devices_is_multi_node(self):
        """Test: 32 GPUs with 16 local devices should be treated as multi-node"""
        # world_size=32, local_world_size=16
        # num_nodes = (32 + 16 - 1) // 16 = 2
        # world_size > local_world_size => is_multi_node=True
        self._setup_parallel_info(
            world_size=32, tp_size=16, ep_size=32, local_world_size=16
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.DECODE

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        # Should treat as multi-node
        self._assert_deepep_config(moe=True, low_latency=True, internode=True)

    def test_prefill_16_gpu_single_node(self):
        """Test: Prefill with 16 GPU single node should not enable low_latency"""
        # world_size=16, local_world_size=16 => single node
        self._setup_parallel_info(
            world_size=16, tp_size=16, ep_size=16, local_world_size=16
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.PREFILL

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        # Prefill should not enable low_latency even on single-node multi-GPU
        self._assert_deepep_config(moe=True, low_latency=False, internode=False)

    def test_inference_16_gpu_single_node(self):
        """Test: Inference with 16 GPU single node should not enable low_latency or internode"""
        # world_size=16, local_world_size=16 => single node
        self._setup_parallel_info(
            world_size=16, tp_size=16, ep_size=16, local_world_size=16
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        # Inference on single-node should not enable low_latency or internode
        self._assert_deepep_config(moe=True, low_latency=False, internode=False)

    def test_tp2_ep4_world8_single_node(self):
        """Test: Decode TP=2, EP=4, world_size=8 on single node with 8 local GPUs"""
        # world_size=8, tp_size=2, ep_size=4, local_world_size=8
        # num_nodes = (8 + 8 - 1) // 8 = 1
        self._setup_parallel_info(
            world_size=8, tp_size=2, ep_size=4, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.DECODE

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        # Single-node multi-GPU with DECODE should enable moe and low_latency
        self._assert_deepep_config(moe=True, low_latency=True, internode=False)

    def test_tp4_ep16_world16_single_node(self):
        """Test: Decode TP=4, EP=16, world_size=16 on single node with 16 local GPUs"""
        # world_size=16, tp_size=4, ep_size=16, local_world_size=16
        # num_nodes = (16 + 16 - 1) // 16 = 1
        self._setup_parallel_info(
            world_size=16, tp_size=4, ep_size=16, local_world_size=16
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.DECODE

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            g_parallel_info=g_parallel_info,
            role_type=role_type,
        )

        # Single-node multi-GPU with DECODE should enable moe and low_latency
        self._assert_deepep_config(moe=True, low_latency=True, internode=False)


if __name__ == "__main__":
    main()
