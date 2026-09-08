py_test (
    name = "deepep_test",
    srcs = ["deepep_test.py"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py/distributed:collective_torch"
    ],
    env = {"GPU_COUNT": "2"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'2'},
)

py_test (
    name = "moriep_test",
    srcs = ["moriep_test.py"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7', 'gpu_count':'8'},
)

# Needs 4 *visible* devices, not 4 devices that can talk directly: every
# collective here goes through NCCL, and the symm_mem fast path in
# collective_torch.all_reduce/all_gather is never eligible because this test
# only ever builds float32 tensors while symm_mem.py:147-149 requires bfloat16.
# Verified on sm86 with peer access unavailable (8x RTX 3090, CUDA_VISIBLE_DEVICES=0-3):
# 4 tests run, 0 skipped, OK -- NCCL falls back to its shared-memory transport.
# The A10 executor presents 4 visible GPUs per node, so gpu_count 4 is
# satisfiable but occupies a whole node. Verification records live with the
# internal CI evidence, not here.
py_test(
    name = "collective_torch_test",
    srcs = ["collective_torch_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    timeout = 'moderate',
    env = {"GPU_COUNT": "4"},
    tags = ["open_skip"],
    exec_properties = {'gpu':'A10', 'gpu_count':'4'},
)

py_test(
    name = "collective_torch_hipgraph_unit_test",
    srcs = ["collective_torch_hipgraph_unit_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    timeout = "short",
)

# Kept on H20: this one needs 4 devices that can address each other, not just 4
# visible devices. allocate_shared_buffer/open_ipc_handle go through
# cudaIpcGetMemHandle + cudaIpcOpenMemHandle(cudaIpcMemLazyEnablePeerAccess)
# (kernels/user_buffer/user_buffers.cu:114, :121-122), and the transfer itself
# dereferences the peer mapping from the device: userbuffers_send memcpys into
# comm->peer_ptr (:38-39) and kuserbuffers_inc atomicAdd_system's a peer flag
# (:7-9, :40) that kuserbuffers_pushrecv spins on (:11-25). Note user_buffers.py
# :118-140 only *warns* when can_device_access_peer is false, so there is no
# guard to fall back on. Measured on sm86 without peer access (8x RTX 3090):
# "CUDA error: peer access is not supported between these two devices",
# 2/2 errors -- it fails loudly rather than skipping, but it does fail. A10 has
# no NVLink and the peer-access/native-peer-atomic status of the A10 CI pool
# could not be established, so moving this is not yet safe.
py_test(
   name = "user_buffers_test",
   srcs = ["user_buffers_test.py"],
   deps = [
       "//rtp_llm:testlib",
       "//rtp_llm/models_py/distributed:user_buffers",
       "//rtp_llm/models_py/distributed:collective_torch"
   ],
   env = {"GPU_COUNT": "4"},
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20', 'gpu_count':'4'},
)

# Needs 4 visible devices, no interconnect: the roundtrip is plain NCCL
# all_gather + reduce_scatter on float32 tensors (cp_moe_allgather_rs_test.py:82-92),
# and float32 disqualifies the symm_mem fast path (symm_mem.py:147-149,
# :183-199 require bfloat16). Passed on sm86 with peer access unavailable
# (8x RTX 3090): 2 tests, OK. A10 CI workers expose 4 GPUs -- see the note on
# collective_torch_test above.
py_test(
    name = "cp_moe_allgather_rs_test",
    srcs = ["cp_moe_allgather_rs_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    timeout = 'moderate',
    env = {"GPU_COUNT": "4"},
    tags = ["open_skip"],
    exec_properties = {'gpu':'A10', 'gpu_count':'4'},
)
