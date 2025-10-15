#include "rtp_llm/cpp/core/MemoryTracker.h"
#include "rtp_llm/cpp/core/TrackerAllocator.h"
#include "rtp_llm/cpp/cuda/allocator_cuda.h"
#include <gtest/gtest.h>

using namespace std;
using namespace rtp_llm;

class MemoryTest: public ::testing::Test {
public:
    void SetUp() {};
    void TearDown() {};
};

TEST_F(MemoryTest, testAlloc) {
    void*         base_ptr = (void*)0xF0000000;
    MemoryTracker tracker(base_ptr, 1 << 20, 1 << 4);
    auto          ptr = tracker.allocate(1 << 10);
    EXPECT_EQ(ptr, base_ptr);
    auto ptr1 = tracker.allocate(1 << 10);
    EXPECT_EQ(ptr1, (char*)base_ptr + (1 << 10));
    tracker.deallocate(ptr);
    auto ptr2 = tracker.allocate(1 << 12);
    EXPECT_EQ(ptr2, (char*)base_ptr + (1 << 11));
    auto ptr3 = tracker.allocate(1 << 8);
    EXPECT_EQ(ptr3, base_ptr);
    auto status = tracker.getStatus();
    auto ptr4   = tracker.allocate((1 << 20) - (1 << 11));
    // cout << status.toString() << endl;
    EXPECT_EQ(ptr4, nullptr);
    EXPECT_EQ(status.available_size, (1 << 20) - (1 << 10) - (1 << 12) - (1 << 8));
    // EXPECT_EQ(status.free_size, (1 << 20) - (1 << 11) - (1 << 12));
    EXPECT_EQ(status.fragmented_size, (1 << 10) - (1 << 8));
    EXPECT_EQ(status.allocated_size, (1 << 10) + (1 << 12) + (1 << 8));
    EXPECT_EQ(status.fragment_chunk_count, 1);
    EXPECT_EQ(status.allocated_chunk_count, 3);

    tracker.deallocate(ptr3);
    tracker.deallocate(ptr1);
    tracker.deallocate(ptr2);
    status = tracker.getStatus();
    // cout << status.toString() << endl;
    EXPECT_EQ(status.available_size, (1 << 20));
    // EXPECT_EQ(status.free_size, (1 << 20));
    EXPECT_EQ(status.fragmented_size, 0);
    EXPECT_EQ(status.allocated_size, 0);
    EXPECT_EQ(status.fragment_chunk_count, 0);
    EXPECT_EQ(status.allocated_chunk_count, 0);
}

TEST_F(MemoryTest, testRandomAlloc) {
    const auto                        test_count = 10000;
    const auto                        align_size = 128;
    MemoryTracker                     tracker((void*)0xF0000000, 1 << 20, align_size);
    std::unordered_map<void*, size_t> ptr_map;

    for (size_t i = 0; i < test_count; i++) {
        size_t hash_val = ((i + 123456789) * 1145141919810 + 114514) * 1919810;
        if (((hash_val >> 60) > 6) || (ptr_map.size() == 0)) {
            auto alloc_size = hash_val & 0xF2;
            if (!alloc_size) {
                continue;
            }
            auto ptr   = tracker.allocate(alloc_size);
            alloc_size = (alloc_size + align_size - 1) / align_size * align_size;
            if (nullptr != ptr) {
                ASSERT_EQ((size_t)ptr % align_size, 0);
                ptr_map[ptr] = alloc_size;
            }
        } else {
            auto ptr = ptr_map.begin()->first;
            tracker.deallocate(ptr);
            ptr_map.erase(ptr);
        }
    }

    auto status = tracker.getStatus();
    // cout << status.toString() << endl;
    EXPECT_EQ(status.allocated_chunk_count, ptr_map.size());
    auto total_size = 0;
    for (auto iter = ptr_map.begin(); iter != ptr_map.end(); iter++) {
        total_size += iter->second;
    }
    EXPECT_EQ(status.allocated_size, total_size);
    EXPECT_EQ(status.available_size, (1 << 20) - total_size);
    // EXPECT_EQ(status.free_size + status.fragmented_size, status.available_size);

    for (auto iter = ptr_map.begin(); iter != ptr_map.end(); iter++) {
        EXPECT_TRUE(tracker.isTracking(iter->first));
        tracker.deallocate(iter->first);
        EXPECT_FALSE(tracker.isTracking(iter->first));
    }
    status = tracker.getStatus();
    // cout << status.toString() << endl;
    EXPECT_EQ(status.allocated_chunk_count, 0);
    EXPECT_EQ(status.allocated_size, 0);
    EXPECT_EQ(status.available_size, (1 << 20));
    // EXPECT_EQ(status.free_size, (1 << 20));
    EXPECT_EQ(status.fragmented_size, 0);
    EXPECT_EQ(status.fragment_chunk_count, 0);
}

TEST_F(MemoryTest, testPrivateAlloc) {
    const auto              align_size   = 128;
    const auto              tracker_size = 1 << 20;  // 1MB
    MemoryTracker           tracker((void*)0xF0000000, tracker_size, align_size);
    std::map<void*, size_t> ptr_map;
    std::map<void*, size_t> private_ptr_map;

    for (size_t i = 0; i < 10; i++) {
        const auto alloc_size = (i + 1) * 4096;  // 4KB, 8KB, ..., 40KB
        auto       ptr        = tracker.allocate(alloc_size);
        ASSERT_NE(ptr, nullptr);
        ASSERT_EQ((size_t)ptr % align_size, 0);
        ptr_map[ptr] = alloc_size;
    }
    for (size_t i = 0; i < 5; i++) {
        auto ptr = ptr_map.begin()->first;
        tracker.deallocate(ptr);
        ptr_map.erase(ptr);
    }

    auto status = tracker.getStatus();
    for (const auto& chunk : status.chunks) {
        printf("Chunk: ptr=%p, size=%zu, used=%d\n", chunk.ptr, chunk.size, chunk.used);
    }
    printf(
        "Tracker Status: available_size=%zu, fragmented_size=%zu, allocated_size=%zu, fragment_chunk_count=%zu, allocated_chunk_count=%zu, allocated_private_size=%zu, freezed_bytes=%zu\n",
        status.available_size,
        status.fragmented_size,
        status.allocated_size,
        status.fragment_chunk_count,
        status.allocated_chunk_count,
        status.allocated_private_size,
        status.freezed_bytes);

    for (size_t i = 0; i < 10; i++) {
        const auto alloc_size = (i + 1) * 8192;  // 8KB, 16KB, ..., 80KB
        auto       ptr        = tracker.allocatePrivate(alloc_size);
        ASSERT_NE(ptr, nullptr);
        ASSERT_EQ((size_t)ptr % align_size, 0);
        private_ptr_map[ptr] = alloc_size;
    }

    status = tracker.getStatus();
    for (const auto& chunk : status.chunks) {
        printf("Chunk: ptr=%p, size=%zu, used=%d\n", chunk.ptr, chunk.size, chunk.used);
    }
    printf(
        "Tracker Status: available_size=%zu, fragmented_size=%zu, allocated_size=%zu, fragment_chunk_count=%zu, allocated_chunk_count=%zu, allocated_private_size=%zu, freezed_bytes=%zu\n",
        status.available_size,
        status.fragmented_size,
        status.allocated_size,
        status.fragment_chunk_count,
        status.allocated_chunk_count,
        status.allocated_private_size,
        status.freezed_bytes);

    EXPECT_EQ(status.allocated_chunk_count, ptr_map.size() + private_ptr_map.size());
    EXPECT_EQ(status.allocated_private_size, 450560);  // 440 KiB
    EXPECT_EQ(status.allocated_size, 614400);          // 440 KiB
    EXPECT_EQ(status.freezed_bytes, 450560);

    for (size_t i = 0; i < 5; i++) {
        auto ptr = private_ptr_map.begin()->first;
        tracker.deallocate(ptr);
        private_ptr_map.erase(ptr);
    }

    for (size_t i = 0; i < 2; i++) {
        auto ptr = private_ptr_map.rbegin()->first;
        tracker.deallocate(ptr);
        private_ptr_map.erase(ptr);
    }

    // this allocation should reuse the freezed memory, not freezing new memory
    for (size_t i = 0; i < 2; i++) {
        auto ptr = tracker.allocatePrivate(3 * 8192);
        ASSERT_NE(ptr, nullptr);
        private_ptr_map[ptr] = 3 * 8192;  // 24KB
    }
    status = tracker.getStatus();
    EXPECT_EQ(status.freezed_bytes, 450560);
    EXPECT_EQ(status.allocated_chunk_count, ptr_map.size() + private_ptr_map.size());
    EXPECT_EQ(status.chunks[6].size, 675840);
    EXPECT_EQ(status.chunks[6].used, false);

    for (const auto& chunk : status.chunks) {
        printf("Chunk: ptr=%p, size=%zu, used=%d\n", chunk.ptr, chunk.size, chunk.used);
    }
    printf(
        "Tracker Status: available_size=%zu, fragmented_size=%zu, allocated_size=%zu, fragment_chunk_count=%zu, allocated_chunk_count=%zu, allocated_private_size=%zu, freezed_bytes=%zu\n",
        status.available_size,
        status.fragmented_size,
        status.allocated_size,
        status.fragment_chunk_count,
        status.allocated_chunk_count,
        status.allocated_private_size,
        status.freezed_bytes);

    {
        auto ptr = tracker.allocate(640000);
        ASSERT_EQ(ptr, nullptr);  // should fail, part of the free block is freezed.
    }

    // free all allocations
    for (auto iter = ptr_map.begin(); iter != ptr_map.end(); iter++) {
        EXPECT_TRUE(tracker.isTracking(iter->first));
        tracker.deallocate(iter->first);
        EXPECT_FALSE(tracker.isTracking(iter->first));
    }
    for (auto iter = private_ptr_map.begin(); iter != private_ptr_map.end(); iter++) {
        EXPECT_TRUE(tracker.isTracking(iter->first));
        tracker.deallocate(iter->first);
        EXPECT_FALSE(tracker.isTracking(iter->first));
    }

    status = tracker.getStatus();
    EXPECT_EQ(status.freezed_bytes, 450560);
    EXPECT_EQ(status.allocated_chunk_count, 0);
    EXPECT_EQ(status.allocated_size, 0);
    EXPECT_EQ(status.available_size, tracker_size);
    EXPECT_EQ(status.fragmented_size, 0);
    EXPECT_EQ(status.fragment_chunk_count, 0);
}

TEST_F(MemoryTest, testVmemAllocate) {
    int  device_id      = 0;
    auto vmem_allocator = new Allocator<AllocatorType::CUDA>(device_id);

    size_t system_free_bytes = 0;
    size_t total_bytes       = 0;
    size_t allocation_size   = 32 * 1024 * 1024;  // 32 MB
    cudaMemGetInfo(&system_free_bytes, &total_bytes);

    auto ptr1 = vmem_allocator->malloc(allocation_size);
    auto ptr2 = vmem_allocator->malloc(allocation_size);
    auto ptr3 = vmem_allocator->mallocPhysical(allocation_size);

    EXPECT_EQ(ptr1 != nullptr, true);
    EXPECT_EQ(ptr2 != nullptr, true);
    EXPECT_EQ(ptr3 != nullptr, true);

    size_t current_free_bytes = 0;
    cudaMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 3 * allocation_size);

    vmem_allocator->free(&ptr1);
    cudaMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 2 * allocation_size);

    vmem_allocator->unmap();
    cudaMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 1 * allocation_size);

    vmem_allocator->free(&ptr2);
    cudaMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 1 * allocation_size);

    vmem_allocator->free(&ptr2);
    cudaMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 1 * allocation_size);

    ptr2 = vmem_allocator->malloc(allocation_size);
    cudaMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 2 * allocation_size);

    vmem_allocator->unmap();
    cudaMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 1 * allocation_size);

    vmem_allocator->map();
    cudaMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 2 * allocation_size);

    vmem_allocator->free(&ptr3);
    cudaMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 1 * allocation_size);

    vmem_allocator->unmap();
    cudaMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 0 * allocation_size);

    vmem_allocator->free(&ptr2);
    cudaMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 0 * allocation_size);
}

TEST_F(MemoryTest, testMemoryTracker) {
    int  device_id                 = 0;
    auto basic_cuda_allocator      = new Allocator<AllocatorType::CUDA>(device_id);
    auto basic_cuda_host_allocator = new Allocator<AllocatorType::CUDA_HOST>(device_id);

    TrackerAllocatorParams params;
    params.real_allocator     = basic_cuda_allocator;
    params.target_track_bytes = 1L * 1024L * 1024L * 1024L;  // 1GB
    params.bytes_try_step     = 128L * 1024L * 1024L;        // 128MB
    params.align_size         = 64;

    TrackerAllocator cuda_allocator(params);
    params.real_allocator = basic_cuda_host_allocator;
    TrackerAllocator cuda_host_allocator(params);

    const auto                test_count = 1000;
    cudaStream_t              stream;
    std::unordered_set<void*> cuda_ptrs;
    std::unordered_set<void*> cuda_host_ptrs;
    for (size_t i = 0; i < test_count; i++) {
        auto alloc_size    = i * 1024;
        auto cuda_ptr      = cuda_allocator.malloc(alloc_size);
        auto cuda_host_ptr = cuda_host_allocator.malloc(alloc_size);
        cuda_ptrs.insert(cuda_ptr);
        cuda_host_ptrs.insert(cuda_host_ptr);
        cudaMemcpyAsync(cuda_ptr, cuda_host_ptr, alloc_size, cudaMemcpyHostToDevice, stream);
    }
    cudaDeviceSynchronize();
    for (auto ptr : cuda_ptrs) {
        cuda_allocator.free(&ptr);
    }
    for (auto ptr : cuda_host_ptrs) {
        cuda_host_allocator.free(&ptr);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
