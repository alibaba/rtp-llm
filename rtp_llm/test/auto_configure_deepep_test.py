from unittest import TestCase, main

from rtp_llm.config.py_config_modules import DeepEPConfig
from rtp_llm.config.server_config_setup import auto_configure_deepep
from rtp_llm.ops import CPRotateMethod, MoeConfig, ParallelismConfig, RoleType


class AutoConfigureDeepepTest(TestCase):
    """Test cases for auto_configure_deepep function"""

    def setUp(self):
        """Set up test environment before each test"""
        # Create fresh config objects for each test
        self.moe_config = MoeConfig()
        self.deep_ep_config = DeepEPConfig()
        self.parallel_config = ParallelismConfig()

    def _setup_parallel_info(
        self,
        world_size: int,
        tp_size: int,
        ep_size: int,
        local_world_size: int,
    ):
        """Set self.parallel_config for auto_configure_deepep."""
        self.parallel_config.world_size = world_size
        self.parallel_config.tp_size = tp_size
        self.parallel_config.ep_size = ep_size
        self.parallel_config.local_world_size = local_world_size

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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
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
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        # Single-node multi-GPU with DECODE should enable moe and low_latency
        self._assert_deepep_config(moe=True, low_latency=True, internode=False)

    def test_tp1_ep4_world4_single_node(self):
        """Test: Decode TP=1, EP=4, world_size=4 on single node with 4 local GPUs"""
        # world_size=4, tp_size=1, ep_size=4, local_world_size=4
        # num_nodes = (4 + 4 - 1) // 4 = 1
        self._setup_parallel_info(
            world_size=4, tp_size=1, ep_size=4, local_world_size=4
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.DECODE

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        # Single-node multi-GPU with DECODE should enable moe and low_latency
        self._assert_deepep_config(moe=True, low_latency=True, internode=False)

    def test_user_specified_all_config_values(self):
        """Test: User explicitly sets all DeepEP configuration values"""
        self._setup_parallel_info(
            world_size=8, tp_size=4, ep_size=8, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        # User explicitly sets all values
        self.deep_ep_config.use_deepep_moe = True
        self.deep_ep_config.use_deepep_low_latency = True
        self.deep_ep_config.use_deepep_internode = True
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        # Should use user-specified values, not auto-config
        self._assert_deepep_config(moe=True, low_latency=True, internode=True)

    def test_user_specified_partial_config_values(self):
        """Test: User explicitly sets only some DeepEP configuration values"""
        self._setup_parallel_info(
            world_size=8, tp_size=4, ep_size=8, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        # User only sets use_deepep_moe
        self.deep_ep_config.use_deepep_moe = False
        self.deep_ep_config.use_deepep_low_latency = None
        self.deep_ep_config.use_deepep_internode = None
        role_type = RoleType.DECODE

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        # Should use user-specified value for moe, and auto-config for others
        # But since user set at least one value, it should copy only the set values
        # Actually, looking at the code, if any value is set, it copies only the set ones
        # and leaves others as they were (or None)
        self.assertEqual(self.moe_config.use_deepep_moe, False)
        # The other values should remain as auto-configured or default

    def test_user_specified_moe_only(self):
        """Test: User explicitly sets only use_deepep_moe"""
        self._setup_parallel_info(
            world_size=8, tp_size=4, ep_size=8, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        self.deep_ep_config.use_deepep_moe = False
        self.deep_ep_config.use_deepep_low_latency = None
        self.deep_ep_config.use_deepep_internode = None
        role_type = RoleType.DECODE

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        # Should use user-specified value
        self.assertEqual(self.moe_config.use_deepep_moe, False)

    def test_user_specified_low_latency_only(self):
        """Test: User explicitly sets only use_deepep_low_latency"""
        self._setup_parallel_info(
            world_size=8, tp_size=4, ep_size=8, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        self.deep_ep_config.use_deepep_moe = None
        self.deep_ep_config.use_deepep_low_latency = True
        self.deep_ep_config.use_deepep_internode = None
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        # Should use user-specified value
        self.assertEqual(self.moe_config.use_deepep_low_latency, True)

    def test_user_specified_internode_only(self):
        """Test: User explicitly sets only use_deepep_internode"""
        self._setup_parallel_info(
            world_size=8, tp_size=4, ep_size=8, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        self.deep_ep_config.use_deepep_moe = None
        self.deep_ep_config.use_deepep_low_latency = None
        self.deep_ep_config.use_deepep_internode = True
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        # Should use user-specified value
        self.assertEqual(self.moe_config.use_deepep_internode, True)

    def test_use_all_gather_disabled_by_low_latency(self):
        """Test: use_deepep_low_latency=True should disable use_all_gather"""
        self._setup_parallel_info(
            world_size=8, tp_size=8, ep_size=8, local_world_size=8
        )
        self.moe_config.use_all_gather = True
        # User sets low_latency, which should disable use_all_gather
        self.deep_ep_config.use_deepep_low_latency = True
        self.deep_ep_config.use_deepep_moe = None
        self.deep_ep_config.use_deepep_internode = None
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        # use_all_gather should be disabled because use_deepep_low_latency is True
        self.assertEqual(self.moe_config.use_all_gather, False)
        # And DeepEP settings should be applied
        self.assertEqual(self.moe_config.use_deepep_low_latency, True)

    def test_use_all_gather_with_ep_not_equal_tp(self):
        """Test: use_all_gather=True but ep_size != tp_size should not disable DeepEP"""
        self._setup_parallel_info(
            world_size=8, tp_size=4, ep_size=8, local_world_size=8
        )
        self.moe_config.use_all_gather = True
        self.deep_ep_config.use_deepep_moe = None
        self.deep_ep_config.use_deepep_low_latency = None
        self.deep_ep_config.use_deepep_internode = None
        role_type = RoleType.DECODE

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        # Since ep_size != tp_size, use_all_gather should be disabled
        # and normal auto-config should apply
        self.assertEqual(self.moe_config.use_all_gather, False)
        # Should apply auto-config for DECODE + single-node multi-GPU
        self._assert_deepep_config(moe=True, low_latency=True, internode=False)

    def test_inference_single_node_multi_gpu(self):
        """Test: Non-PD separation + Inference node + Single-node multi-GPU (>1TP): 1, 0, 0"""
        self._setup_parallel_info(
            world_size=8, tp_size=4, ep_size=8, local_world_size=8
        )
        self.moe_config.use_all_gather = False
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        self._assert_deepep_config(moe=True, low_latency=False, internode=False)

    def test_deepep_ll_init(self):
        """Test: deepep ll max token init"""
        self._setup_parallel_info(
            world_size=8, tp_size=4, ep_size=8, local_world_size=8
        )
        role_type = RoleType.PDFUSION
        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
            ll_num_max_token=256,
        )
        self.assertEqual(self.moe_config.ll_num_max_token, 256)

    # --- CP+EP auto configuration tests ---
    # Topology: tp_size > 1, dp_size == 1, ep_size == tp_size with CP enabled.
    # Shape matches pure TP, but CP is on. Must NOT be classified as pure TP:
    # use_all_gather should be cleared and DeepEP auto-config should kick in.
    # Opting into PureCP allgather+RS still requires explicit --moe_strategy.

    def test_cp_plus_ep_inference_falls_back_to_deepep(self):
        """CP enabled with tp==ep==2 (PDFUSION) must NOT match is_pure_tp."""
        self._setup_parallel_info(
            world_size=2, tp_size=2, ep_size=2, local_world_size=8
        )
        self.parallel_config.dp_size = 1
        self.parallel_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        self.moe_config.use_all_gather = True  # default-on flag from MoeConfig
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        # CP enabled => pure TP path skipped => use_all_gather cleared
        self.assertFalse(self.moe_config.use_all_gather)
        # Inference + single-node multi-GPU => DeepEP on
        self._assert_deepep_config(moe=True, low_latency=False, internode=False)

    def test_cp_plus_ep_prefill_falls_back_to_deepep(self):
        """CP enabled with tp==ep==2 in PD-separation PREFILL also falls back."""
        self._setup_parallel_info(
            world_size=2, tp_size=2, ep_size=2, local_world_size=8
        )
        self.parallel_config.dp_size = 1
        self.parallel_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        self.moe_config.use_all_gather = True
        role_type = RoleType.PREFILL

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        self.assertFalse(self.moe_config.use_all_gather)
        # PD prefill + single-node multi-GPU => DeepEP on, low_latency off
        self._assert_deepep_config(moe=True, low_latency=False, internode=False)

    def test_cp_plus_ep_alltoall_method_also_falls_back(self):
        """Any non-DISABLED/PREFILL_CP/UNKNOWN method counts as CP enabled."""
        self._setup_parallel_info(
            world_size=2, tp_size=2, ep_size=2, local_world_size=8
        )
        self.parallel_config.dp_size = 1
        self.parallel_config.prefill_cp_config.method = CPRotateMethod.ALLTOALL
        self.moe_config.use_all_gather = True
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        self.assertFalse(self.moe_config.use_all_gather)
        self._assert_deepep_config(moe=True, low_latency=False, internode=False)

    def test_pure_tp_without_cp_still_uses_allgather(self):
        """Regression guard: same shape (tp==ep==2, dp==1) but CP DISABLED is pure TP."""
        self._setup_parallel_info(
            world_size=2, tp_size=2, ep_size=2, local_world_size=8
        )
        self.parallel_config.dp_size = 1
        self.parallel_config.prefill_cp_config.method = CPRotateMethod.DISABLED
        self.moe_config.use_all_gather = True
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        # Pure TP path: use_all_gather stays on, DeepEP all disabled
        self.assertTrue(self.moe_config.use_all_gather)
        self._assert_deepep_config(moe=False, low_latency=False, internode=False)

    def test_prefill_cp_method_is_not_treated_as_cp_enabled(self):
        """CPRotateMethod.PREFILL_CP must not flip is_pure_tp off (is_enabled() returns False)."""
        self._setup_parallel_info(
            world_size=2, tp_size=2, ep_size=2, local_world_size=8
        )
        self.parallel_config.dp_size = 1
        self.parallel_config.prefill_cp_config.method = CPRotateMethod.PREFILL_CP
        self.moe_config.use_all_gather = True
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        # PREFILL_CP method is not "is_enabled()" per ConfigModules.h, so pure TP path holds
        self.assertTrue(self.moe_config.use_all_gather)
        self._assert_deepep_config(moe=False, low_latency=False, internode=False)

    # --- Explicit --moe_strategy opt-in tests ---
    # use_all_gather must survive auto_configure_deepep when the user explicitly
    # opts into the PureCP / PureDP allgather+RS routers, otherwise the matching
    # strategy's check_conditions (which requires use_all_gather) never holds.

    def test_explicit_pure_dp_preserves_allgather(self):
        """--moe_strategy=fp8_per_block_pure_dp on DP+EP topology keeps use_all_gather."""
        self._setup_parallel_info(
            world_size=2, tp_size=1, ep_size=2, local_world_size=8
        )
        self.parallel_config.dp_size = 2
        self.moe_config.use_all_gather = True
        self.moe_config.moe_strategy = "fp8_per_block_pure_dp"
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        self.assertTrue(self.moe_config.use_all_gather)
        self._assert_deepep_config(moe=False, low_latency=False, internode=False)

    def test_explicit_pure_cp_preserves_allgather(self):
        """--moe_strategy=fp8_per_block_pure_cp on CP+EP topology keeps use_all_gather."""
        self._setup_parallel_info(
            world_size=2, tp_size=2, ep_size=2, local_world_size=8
        )
        self.parallel_config.dp_size = 1
        self.parallel_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        self.moe_config.use_all_gather = True
        self.moe_config.moe_strategy = "fp8_per_block_pure_cp"
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        self.assertTrue(self.moe_config.use_all_gather)
        self._assert_deepep_config(moe=False, low_latency=False, internode=False)

    def test_explicit_pure_dp_wrong_topology_falls_back(self):
        """--moe_strategy=fp8_per_block_pure_dp on mixed tp+dp must NOT preserve use_all_gather."""
        # tp>1 fails explicit_pure_dp (requires tp==1); dp>1 fails is_pure_tp (requires dp==1).
        # Neither allow-list condition matches => use_all_gather cleared, DeepEP auto-selected.
        self._setup_parallel_info(
            world_size=4, tp_size=2, ep_size=4, local_world_size=8
        )
        self.parallel_config.dp_size = 2
        self.moe_config.use_all_gather = True
        self.moe_config.moe_strategy = "fp8_per_block_pure_dp"
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        self.assertFalse(self.moe_config.use_all_gather)
        self._assert_deepep_config(moe=True, low_latency=False, internode=False)

    def test_single_gpu_with_use_deepep_moe_keeps_allgather(self):
        """Single GPU + --use_deepep_moe 1 must keep use_all_gather (no DeepEP path on 1 GPU).

        Regression guard for moe_headwise smoke (world_size=1, tp_size=1, ep_size=1
        with --use_deepep_moe 1): use_deepep_moe must NOT silently disable
        use_all_gather on a single GPU, otherwise no MoE strategy matches.
        """
        self._setup_parallel_info(
            world_size=1, tp_size=1, ep_size=1, local_world_size=8
        )
        self.deep_ep_config.use_deepep_moe = True
        self.moe_config.use_all_gather = True
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        self.assertTrue(self.moe_config.use_all_gather)
        # use_all_gather wins => DeepEP forcibly cleared
        self._assert_deepep_config(moe=False, low_latency=False, internode=False)

    def test_multi_gpu_use_deepep_moe_disables_allgather(self):
        """Multi-GPU + --use_deepep_moe 1 --use_deepep_low_latency 0: use_all_gather cleared."""
        # Mirror the CLI shape: user sets both DeepEP flags explicitly so the
        # else-branch copies both onto moe_config; otherwise use_deepep_low_latency
        # keeps its MoeConfig default (True).
        self._setup_parallel_info(
            world_size=2, tp_size=2, ep_size=2, local_world_size=8
        )
        self.parallel_config.dp_size = 1
        self.deep_ep_config.use_deepep_moe = True
        self.deep_ep_config.use_deepep_low_latency = False
        self.moe_config.use_all_gather = True
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        self.assertFalse(self.moe_config.use_all_gather)
        self.assertTrue(self.moe_config.use_deepep_moe)
        self.assertFalse(self.moe_config.use_deepep_low_latency)

    def test_auto_strategy_does_not_preserve_allgather_on_dp(self):
        """Regression: moe_strategy=auto on DP topology must clear use_all_gather."""
        self._setup_parallel_info(
            world_size=2, tp_size=1, ep_size=2, local_world_size=8
        )
        self.parallel_config.dp_size = 2
        self.moe_config.use_all_gather = True
        self.moe_config.moe_strategy = "auto"
        role_type = RoleType.PDFUSION

        auto_configure_deepep(
            moe_config=self.moe_config,
            deep_ep_config=self.deep_ep_config,
            parallelism_config=self.parallel_config,
            role_type=role_type,
        )

        self.assertFalse(self.moe_config.use_all_gather)
        self._assert_deepep_config(moe=True, low_latency=False, internode=False)


if __name__ == "__main__":
    main()
