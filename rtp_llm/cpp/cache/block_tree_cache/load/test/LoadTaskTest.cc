#include <cstdlib>
#include <new>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTask.h"

namespace {
thread_local size_t                                    last_descriptor_block_bytes = 0;
thread_local size_t                                    remaining_descriptor_copies = 0;
thread_local rtp_llm::TransferTask::Clock::time_point* descriptor_assembly_time    = nullptr;
}  // namespace

// Keep allocation interposition in this dedicated test binary.
[[gnu::noinline]] void* operator new(std::size_t size) {
    if (descriptor_assembly_time) {
        if (remaining_descriptor_copies == 0) {
            // The next allocation follows completion of the final descriptor's two block-list copies.
            *descriptor_assembly_time = rtp_llm::TransferTask::Clock::now();
            descriptor_assembly_time  = nullptr;
        } else if (size == last_descriptor_block_bytes) {
            --remaining_descriptor_copies;
        }
    }
    if (void* pointer = std::malloc(size ? size : 1)) {
        return pointer;
    }
    throw std::bad_alloc();
}
[[gnu::noinline]] void* operator new[](std::size_t size) {
    return ::operator new(size);
}
[[gnu::noinline]] void operator delete(void* pointer) noexcept {
    std::free(pointer);
}
[[gnu::noinline]] void operator delete[](void* pointer) noexcept {
    std::free(pointer);
}
[[gnu::noinline]] void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}
[[gnu::noinline]] void operator delete[](void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

namespace rtp_llm {
namespace {

TEST(LoadTaskTest, DeadlinesStartAfterLastDescriptorIsAssembled) {
    auto coordinator = std::make_shared<LoadContextCoordinator>([](const auto&) { return true; }, [](auto&) {});
    const BlockIdList last_blocks(257, 2);
    auto              context = coordinator->create(
        {TransferDescriptor::hostToDevice(0, 1, {1}), TransferDescriptor::diskToDevice(0, 2, last_blocks)},
        {false, false},
        {},
        2);
    TransferTask::Clock::time_point assembled_at;
    last_descriptor_block_bytes = last_blocks.size() * sizeof(BlockIdxType);
    remaining_descriptor_copies = 2;
    descriptor_assembly_time    = &assembled_at;
    const auto next_trace_id    = TransferTask::next_trace_id_.load();
    auto       task             = LoadTask::create(context, 10, 20);
    descriptor_assembly_time    = nullptr;
    EXPECT_EQ(TransferTask::next_trace_id_.load() - next_trace_id, 2u);
    ASSERT_EQ(remaining_descriptor_copies, 0u);
    ASSERT_NE(assembled_at, TransferTask::Clock::time_point{});
    EXPECT_EQ(task->load_descs.back().target_blocks, last_blocks);
    EXPECT_EQ(task->disk_to_device_task.descriptors().back().target_blocks, last_blocks);
    EXPECT_GE(task->host_to_device_task.deadline() - std::chrono::milliseconds(10), assembled_at);
    EXPECT_GE(task->disk_to_device_task.deadline() - std::chrono::milliseconds(20), assembled_at);
}

}  // namespace
}  // namespace rtp_llm
