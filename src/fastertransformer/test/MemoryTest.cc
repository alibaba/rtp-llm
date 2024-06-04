#include "src/fastertransformer/core/MemoryTracker.h"
#include "src/fastertransformer/core/TrackerAllocator.h"
#include "src/fastertransformer/cuda/allocator_cuda.h"
#include <gtest/gtest.h>

using namespace std;
using namespace fastertransformer;

class MemoryTest: public ::testing::Test {
public:
    void SetUp() {};
    void TearDown() {};
};

TEST_F(MemoryTest, testAlloc) {
    void* base_ptr = (void *)0xF0000000;
    MemoryTracker tracker(base_ptr, 1 << 20, 1 << 4);
    auto ptr = tracker.allocate(1 << 10);
    EXPECT_EQ(ptr, base_ptr);
    auto ptr1 = tracker.allocate(1 << 10);
    EXPECT_EQ(ptr1, base_ptr + (1 << 10));
    tracker.deallocate(ptr);
    auto ptr2 = tracker.allocate(1 << 12);
    EXPECT_EQ(ptr2, base_ptr + (1 << 11));
    auto ptr3 = tracker.allocate(1 << 8);
    EXPECT_EQ(ptr3, base_ptr);
    auto status = tracker.getStatus();
    auto ptr4 = tracker.allocate((1 << 20) - (1 << 11));
    // cout << status.toString() << endl;
    EXPECT_EQ(ptr4, nullptr);
    EXPECT_EQ(status.available_size, (1 << 20) - (1 << 10) - (1 << 12) - (1 << 8));
    EXPECT_EQ(status.free_size, (1 << 20) - (1 << 11) - (1 << 12));
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
    EXPECT_EQ(status.free_size, (1 << 20));
    EXPECT_EQ(status.fragmented_size, 0);
    EXPECT_EQ(status.allocated_size, 0);
    EXPECT_EQ(status.fragment_chunk_count, 0);
    EXPECT_EQ(status.allocated_chunk_count, 0);
}

TEST_F(MemoryTest, testRandomAlloc) {
    const auto test_count = 10000;
    const auto align_size = 128;
    MemoryTracker tracker((void *)0xF0000000, 1 << 20, align_size);
    std::unordered_map<void*, size_t> ptr_map;

    for (size_t i = 0; i < test_count; i++) {
        size_t hash_val = ((i + 123456789) * 1145141919810 + 114514) * 1919810;
        if (((hash_val >> 60) > 6) || (ptr_map.size() == 0)) {
            auto alloc_size = hash_val & 0xF2;
            auto ptr = tracker.allocate(alloc_size);
            alloc_size = (alloc_size + align_size - 1) / align_size * align_size;
            if (nullptr != ptr) {
                ASSERT_EQ((size_t)ptr % align_size, 0);
                ptr_map[ptr] = alloc_size;
            }
        } else {
            auto idx = hash_val % ptr_map.size();
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
    EXPECT_EQ(status.free_size + status.fragmented_size, status.available_size);

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
    EXPECT_EQ(status.free_size, (1 << 20));
    EXPECT_EQ(status.fragmented_size, 0);
    EXPECT_EQ(status.fragment_chunk_count, 0);
}

TEST_F(MemoryTest, testMemoryTracker) {
    int device_id = 0;
    auto basic_cuda_allocator = new Allocator<AllocatorType::CUDA>(device_id);
    auto basic_cuda_host_allocator = new Allocator<AllocatorType::CUDA_HOST>(device_id);

    TrackerAllocatorParams params;
    params.real_allocator = basic_cuda_allocator;
    params.target_track_bytes = 1L * 1024L * 1024L * 1024L; // 1GB
    params.bytes_try_step = 128L * 1024L * 1024L;          // 128MB
    params.align_size = 64;

    TrackerAllocator cuda_allocator(params);
    params.real_allocator = basic_cuda_host_allocator;
    TrackerAllocator cuda_host_allocator(params);

    const auto test_count = 1000;
    cudaStream_t stream;
    std::unordered_set<void*> cuda_ptrs;
    std::unordered_set<void*> cuda_host_ptrs;
    for (size_t i = 0; i < test_count; i++) {
        auto alloc_size = i * 1024;
        auto cuda_ptr = cuda_allocator.malloc(alloc_size);
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

