"""Public CUDA 13 x86 pins for the GLM-5.3 build."""

# Open builds use the shared HC, selected-KV MLA and indexer paths.
CUDA_EXTENSION = None
MHC_SOURCE = None

DEEP_GEMM_WHEEL = struct(
    url = "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/cu130/deep_gemm-2.8.2%2B522942a.cu132-cp310-cp310-linux_x86_64.whl",
    sha256 = "806896fa0e88c5c50f5f02bc1c0c72983506a9bc079bed4d6829b184e7e83957",
)
FLASH_MLA_REQUIREMENT = None
LINEAR_ATTENTION_REQUIREMENTS = []
