#include "src/fastertransformer/core/MemoryTracker.h"
#include <gtest/gtest.h>

using namespace std;
using namespace fastertransformer;

class LoraGemmTest: public ::testing::Test {
public:
    void SetUp() {};
    void TearDown() {};
};

TEST_F(LoraGemmTest, testAlloc) {
    void* base_ptr = (void *)0xF0000000;
    MemoryTracker tracker(base_ptr, 1 << 20);
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
    cout << tracker.getAllocationInfo() << endl;
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
    cout << tracker.getAllocationInfo() << endl;
    EXPECT_EQ(status.available_size, (1 << 20));
    EXPECT_EQ(status.free_size, (1 << 20));
    EXPECT_EQ(status.fragmented_size, 0);
    EXPECT_EQ(status.allocated_size, 0);
    EXPECT_EQ(status.fragment_chunk_count, 0);
    EXPECT_EQ(status.allocated_chunk_count, 0);
}

TEST_F(LoraGemmTest, testRandomAlloc) {
    const auto test_count = 10000;
    std::unordered_map<void*, size_t> ptr_map;
    MemoryTracker tracker((void *)0xF0000000, 1 << 20);

    for (size_t i = 0; i < test_count; i++) {
        size_t hash_val = ((i + 123456789) * 1145141919810 + 114514) * 1919810;
        if (((hash_val >> 60) > 6) || (ptr_map.size() == 0)) {
            auto alloc_size = hash_val & 0xF0;
            auto ptr = tracker.allocate(alloc_size);
            if (nullptr != ptr) {
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
    cout << tracker.getAllocationInfo() << endl;
    EXPECT_EQ(status.allocated_chunk_count, ptr_map.size());
    auto total_size = 0;
    for (auto iter = ptr_map.begin(); iter != ptr_map.end(); iter++) {
        total_size += iter->second;
    }
    EXPECT_EQ(status.allocated_size, total_size);
    EXPECT_EQ(status.available_size, (1 << 20) - total_size);
    EXPECT_EQ(status.free_size + status.fragmented_size, status.available_size);

    for (auto iter = ptr_map.begin(); iter != ptr_map.end(); iter++) {
        tracker.deallocate(iter->first);
    }
    status = tracker.getStatus();
    cout << tracker.getAllocationInfo() << endl;
    EXPECT_EQ(status.allocated_chunk_count, 0);
    EXPECT_EQ(status.allocated_size, 0);
    EXPECT_EQ(status.available_size, (1 << 20));
    EXPECT_EQ(status.free_size, (1 << 20));
    EXPECT_EQ(status.fragmented_size, 0);
    EXPECT_EQ(status.fragment_chunk_count, 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

