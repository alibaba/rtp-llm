import pathlib
import re
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


class WritebackAllocatorApiStaticTest(unittest.TestCase):
    def test_writeback_allocation_api_is_wired_to_all_allocators(self):
        allocator_h = (REPO_ROOT / "rtp_llm/cpp/cache/KVCacheAllocator.h").read_text()
        manager_h = (REPO_ROOT / "rtp_llm/cpp/cache/KVCacheManager.h").read_text()
        manager_cc = (REPO_ROOT / "rtp_llm/cpp/cache/KVCacheManager.cc").read_text()
        single_h = (
            REPO_ROOT / "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
        ).read_text()
        single_cc = (
            REPO_ROOT / "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.cc"
        ).read_text()
        hybrid_h = (
            REPO_ROOT / "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"
        ).read_text()
        hybrid_cc = (
            REPO_ROOT / "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.cc"
        ).read_text()

        self.assertRegex(
            allocator_h, re.compile(r"virtual\s+absl::Status\s+mallocWritebackBlocks")
        )
        self.assertRegex(
            allocator_h, re.compile(r"virtual\s+void\s+commitWritebackBlocks")
        )

        self.assertRegex(manager_h, re.compile(r"absl::Status\s+mallocWritebackBlocks"))
        self.assertRegex(manager_h, re.compile(r"void\s+commitWritebackBlocks"))
        self.assertIn("allocator_->mallocWritebackBlocks", manager_cc)
        self.assertIn("start_block_index", manager_h)
        self.assertIn("start_block_index", manager_cc)
        self.assertIn("allocator_->commitWritebackBlocks", manager_cc)

        self.assertIn("mallocWritebackBlocks", single_h)
        self.assertIn("commitWritebackBlocks", single_h)
        self.assertIn("SingleTypeKVCacheAllocator::mallocWritebackBlocks", single_cc)
        self.assertIn("SingleTypeKVCacheAllocator::commitWritebackBlocks", single_cc)

        self.assertIn("mallocWritebackBlocks", hybrid_h)
        self.assertIn("commitWritebackBlocks", hybrid_h)
        self.assertIn("HybridTypeKVCacheAllocator::mallocWritebackBlocks", hybrid_cc)
        self.assertIn("HybridTypeKVCacheAllocator::commitWritebackBlocks", hybrid_cc)


if __name__ == "__main__":
    unittest.main()
