#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/KVCachePhysicalMemoryController.h"

namespace rtp_llm {
namespace test {

// CPU-only mock backend: records pause/resume calls and allows failure injection.
class MockBackend: public PhysicalMemoryBackend {
public:
    bool isAvailable() const override {
        return available_;
    }
    std::string name() const override {
        return "mock";
    }
    bool pause(const std::string& tag) override {
        pause_calls_.push_back(tag);
        return !fail_pause_;
    }
    bool resume(const std::string& tag) override {
        resume_calls_.push_back(tag);
        return !fail_resume_;
    }

    bool                     available_   = true;
    bool                     fail_pause_  = false;
    bool                     fail_resume_ = false;
    std::vector<std::string> pause_calls_;
    std::vector<std::string> resume_calls_;
};

class KVCachePhysicalMemoryControllerTest: public ::testing::Test {
protected:
    void SetUp() override {
        backend_    = std::make_shared<MockBackend>();
        controller_ = std::make_unique<KVCachePhysicalMemoryController>(backend_, "kv_cache");
    }

    // A fake device pointer; the controller never dereferences it in attach mode.
    void* fakePtr() {
        return reinterpret_cast<void*>(0xdead0000);
    }

    std::shared_ptr<MockBackend>                     backend_;
    std::unique_ptr<KVCachePhysicalMemoryController> controller_;
};

TEST_F(KVCachePhysicalMemoryControllerTest, AttachRecordsPtrAndSize) {
    EXPECT_EQ(controller_->basePtr(), nullptr);
    EXPECT_EQ(controller_->totalSizeBytes(), 0u);

    void* attached = controller_->allocateOrAttach(fakePtr(), 4096);
    EXPECT_EQ(attached, fakePtr());
    EXPECT_EQ(controller_->basePtr(), fakePtr());
    EXPECT_EQ(controller_->totalSizeBytes(), 4096u);
    EXPECT_FALSE(controller_->isPaused());
}

TEST_F(KVCachePhysicalMemoryControllerTest, AttachValidation) {
    // invalid buffer
    EXPECT_EQ(controller_->allocateOrAttach(nullptr, 4096), nullptr);
    EXPECT_EQ(controller_->allocateOrAttach(fakePtr(), 0), nullptr);

    ASSERT_EQ(controller_->allocateOrAttach(fakePtr(), 4096), fakePtr());
    // idempotent re-attach with identical {ptr, size}
    EXPECT_EQ(controller_->allocateOrAttach(fakePtr(), 4096), fakePtr());
    // conflicting re-attach rejected
    EXPECT_EQ(controller_->allocateOrAttach(reinterpret_cast<void*>(0xbeef0000), 4096), nullptr);
    EXPECT_EQ(controller_->allocateOrAttach(fakePtr(), 8192), nullptr);
    // original attachment untouched
    EXPECT_EQ(controller_->basePtr(), fakePtr());
    EXPECT_EQ(controller_->totalSizeBytes(), 4096u);
}

TEST_F(KVCachePhysicalMemoryControllerTest, PauseThenResume) {
    ASSERT_EQ(controller_->allocateOrAttach(fakePtr(), 4096), fakePtr());

    EXPECT_TRUE(controller_->pausePhysicalMemory());
    EXPECT_TRUE(controller_->isPaused());
    ASSERT_EQ(backend_->pause_calls_.size(), 1u);
    EXPECT_EQ(backend_->pause_calls_[0], "kv_cache");

    EXPECT_TRUE(controller_->resumePhysicalMemory());
    EXPECT_FALSE(controller_->isPaused());
    ASSERT_EQ(backend_->resume_calls_.size(), 1u);
    EXPECT_EQ(backend_->resume_calls_[0], "kv_cache");
}

TEST_F(KVCachePhysicalMemoryControllerTest, BasePtrStableAcrossPauseResumeCycles) {
    ASSERT_EQ(controller_->allocateOrAttach(fakePtr(), 4096), fakePtr());

    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(controller_->pausePhysicalMemory());
        EXPECT_EQ(controller_->basePtr(), fakePtr());
        EXPECT_EQ(controller_->totalSizeBytes(), 4096u);
        EXPECT_TRUE(controller_->resumePhysicalMemory());
        EXPECT_EQ(controller_->basePtr(), fakePtr());
        EXPECT_EQ(controller_->totalSizeBytes(), 4096u);
    }
}

TEST_F(KVCachePhysicalMemoryControllerTest, RepeatedPauseAndResumeAreIdempotent) {
    ASSERT_EQ(controller_->allocateOrAttach(fakePtr(), 4096), fakePtr());

    // resume while running: no-op success, backend untouched
    EXPECT_TRUE(controller_->resumePhysicalMemory());
    EXPECT_TRUE(backend_->resume_calls_.empty());

    EXPECT_TRUE(controller_->pausePhysicalMemory());
    // second pause while paused: no-op success, backend called only once
    EXPECT_TRUE(controller_->pausePhysicalMemory());
    EXPECT_TRUE(controller_->isPaused());
    EXPECT_EQ(backend_->pause_calls_.size(), 1u);

    EXPECT_TRUE(controller_->resumePhysicalMemory());
    // second resume while running: no-op success, backend called only once
    EXPECT_TRUE(controller_->resumePhysicalMemory());
    EXPECT_FALSE(controller_->isPaused());
    EXPECT_EQ(backend_->resume_calls_.size(), 1u);
}

TEST_F(KVCachePhysicalMemoryControllerTest, PauseWithoutAttachFails) {
    EXPECT_FALSE(controller_->pausePhysicalMemory());
    EXPECT_FALSE(controller_->isPaused());
    EXPECT_TRUE(backend_->pause_calls_.empty());
}

TEST_F(KVCachePhysicalMemoryControllerTest, AttachWhilePausedFails) {
    ASSERT_EQ(controller_->allocateOrAttach(fakePtr(), 4096), fakePtr());
    ASSERT_TRUE(controller_->pausePhysicalMemory());
    EXPECT_EQ(controller_->allocateOrAttach(fakePtr(), 4096), nullptr);
}

TEST_F(KVCachePhysicalMemoryControllerTest, UnavailableBackendFails) {
    backend_->available_ = false;
    EXPECT_FALSE(controller_->backendAvailable());

    ASSERT_EQ(controller_->allocateOrAttach(fakePtr(), 4096), fakePtr());
    EXPECT_FALSE(controller_->pausePhysicalMemory());
    EXPECT_FALSE(controller_->isPaused());
    EXPECT_TRUE(backend_->pause_calls_.empty());
}

TEST_F(KVCachePhysicalMemoryControllerTest, NullBackendFails) {
    KVCachePhysicalMemoryController controller(nullptr);
    EXPECT_FALSE(controller.backendAvailable());
    ASSERT_EQ(controller.allocateOrAttach(fakePtr(), 4096), fakePtr());
    EXPECT_FALSE(controller.pausePhysicalMemory());
    EXPECT_FALSE(controller.isPaused());
}

TEST_F(KVCachePhysicalMemoryControllerTest, BackendPauseFailureKeepsRunningState) {
    ASSERT_EQ(controller_->allocateOrAttach(fakePtr(), 4096), fakePtr());

    backend_->fail_pause_ = true;
    EXPECT_FALSE(controller_->pausePhysicalMemory());
    EXPECT_FALSE(controller_->isPaused());

    // recovers once the backend succeeds
    backend_->fail_pause_ = false;
    EXPECT_TRUE(controller_->pausePhysicalMemory());
    EXPECT_TRUE(controller_->isPaused());
}

TEST_F(KVCachePhysicalMemoryControllerTest, BackendResumeFailureKeepsPausedState) {
    ASSERT_EQ(controller_->allocateOrAttach(fakePtr(), 4096), fakePtr());
    ASSERT_TRUE(controller_->pausePhysicalMemory());

    backend_->fail_resume_ = true;
    EXPECT_FALSE(controller_->resumePhysicalMemory());
    EXPECT_TRUE(controller_->isPaused());

    backend_->fail_resume_ = false;
    EXPECT_TRUE(controller_->resumePhysicalMemory());
    EXPECT_FALSE(controller_->isPaused());
}

TEST_F(KVCachePhysicalMemoryControllerTest, CustomTagIsForwardedToBackend) {
    KVCachePhysicalMemoryController controller(backend_, "my_tag");
    ASSERT_EQ(controller.allocateOrAttach(fakePtr(), 4096), fakePtr());
    EXPECT_TRUE(controller.pausePhysicalMemory());
    EXPECT_TRUE(controller.resumePhysicalMemory());
    ASSERT_EQ(backend_->pause_calls_.size(), 1u);
    ASSERT_EQ(backend_->resume_calls_.size(), 1u);
    EXPECT_EQ(backend_->pause_calls_[0], "my_tag");
    EXPECT_EQ(backend_->resume_calls_[0], "my_tag");
}

// VmmBackend dlsym probe: this test binary is not started with the torch_memory_saver
// LD_PRELOAD shim, so the probe must cleanly report "unavailable" instead of crashing.
TEST(VmmBackendTest, UnavailableWithoutPreloadShim) {
    VmmBackend backend;
    EXPECT_FALSE(backend.isAvailable());
    EXPECT_EQ(backend.name(), "vmm");
    // Allocation tagging is a strictly stronger capability than pause/resume; with no shim at all
    // it must also report unavailable, and beginAllocationRegion must refuse rather than pretend.
    EXPECT_FALSE(backend.supportsAllocationTagging());
    EXPECT_FALSE(backend.pause("kv_cache"));
    EXPECT_FALSE(backend.resume("kv_cache"));
    EXPECT_FALSE(backend.beginAllocationRegion("kv_cache"));
    EXPECT_FALSE(backend.endAllocationRegion());

    // A controller on top of an unavailable VMM backend degrades gracefully.
    KVCachePhysicalMemoryController controller(std::make_shared<VmmBackend>());
    ASSERT_EQ(controller.allocateOrAttach(reinterpret_cast<void*>(0xdead0000), 4096),
              reinterpret_cast<void*>(0xdead0000));
    EXPECT_FALSE(controller.pausePhysicalMemory());
    EXPECT_FALSE(controller.isPaused());
}

}  // namespace test
}  // namespace rtp_llm
