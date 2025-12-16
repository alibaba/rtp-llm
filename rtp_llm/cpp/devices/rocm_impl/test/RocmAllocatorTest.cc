#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmAllocator.h"

using namespace std;
using namespace rtp_llm;

class ROCmMemoryTest: public DeviceTestBase {};

TEST_F(ROCmMemoryTest, testVmemAllocate) {
    auto vmem_allocator = new Allocator<AllocatorType::ROCM>();

    size_t system_free_bytes = 0;
    size_t total_bytes       = 0;
    size_t allocation_size   = 32 * 1024 * 1024;  // 32 MB
    hipMemGetInfo(&system_free_bytes, &total_bytes);

    auto ptr1 = vmem_allocator->malloc(allocation_size);
    auto ptr2 = vmem_allocator->malloc(allocation_size);
    auto ptr3 = vmem_allocator->mallocResidentMemory(allocation_size);

    EXPECT_EQ(ptr1 != nullptr, true);
    EXPECT_EQ(ptr2 != nullptr, true);
    EXPECT_EQ(ptr3 != nullptr, true);

    size_t current_free_bytes = 0;
    hipMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 3 * allocation_size);

    vmem_allocator->free(&ptr1);
    hipMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 2 * allocation_size);

    vmem_allocator->unmap();
    hipMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 1 * allocation_size);

    vmem_allocator->free(&ptr2);
    hipMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 1 * allocation_size);

    vmem_allocator->free(&ptr2);
    hipMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 1 * allocation_size);

    ptr2 = vmem_allocator->malloc(allocation_size);
    hipMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 2 * allocation_size);

    vmem_allocator->unmap();
    hipMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 1 * allocation_size);

    vmem_allocator->map();
    hipMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 2 * allocation_size);

    vmem_allocator->free(&ptr3);
    hipMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 1 * allocation_size);

    vmem_allocator->unmap();
    hipMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 0 * allocation_size);

    vmem_allocator->free(&ptr2);
    hipMemGetInfo(&current_free_bytes, &total_bytes);
    EXPECT_EQ(system_free_bytes - current_free_bytes, 0 * allocation_size);
}