#include "rtp_llm/cpp/cache/ThreeFSMempool.h"

#include <iostream>
#include <vector>

namespace rtp_llm::threefs {
void mempoolTest() {
    const size_t         buffer_size = 1024;
    std::vector<uint8_t> buffer(buffer_size);

    ThreeFSMempool pool(buffer.data(), buffer_size);
    if (!pool.init()) {
        std::cout << "memory pool init failed" << std::endl;
        return;
    }

    std::cout << "Initial state:";
    pool.printStatus();

    void* p1 = pool.alloc(128);
    std::cout << "alloc 128 bytes: " << p1 << std::endl;

    void* p2 = pool.alloc(256);
    std::cout << "alloc 256 bytes: " << p2 << std::endl;

    void* p3 = pool.alloc(64);
    std::cout << "alloc 64 bytes: " << p3 << std::endl;

    std::cout << "\nAfter allocations:";
    pool.printStatus();

    pool.free(p2);
    std::cout << "\nAfter freeing middle block:";
    pool.printStatus();

    void* p4 = pool.alloc(192);
    std::cout << "alloc 192 bytes: " << p4 << std::endl;
    std::cout << "\nAfter allocating new block:";
    pool.printStatus();

    void* p5 = pool.alloc(512);
    std::cout << "alloc 512 bytes: " << p5 << std::endl;
    pool.printStatus();

    void* p6 = pool.alloc(512);  // 应该失败
    std::cout << "alloc 512 bytes: " << p6 << std::endl;
    pool.printStatus();

    pool.free(p4);
    std::cout << "\nAfter freeing 192 bytes:";
    pool.printStatus();

    pool.free(p3);
    std::cout << "\nAfter freeing 64 bytes:";
    pool.printStatus();

    pool.free(p1);
    std::cout << "\nAfter freeing 128 bytes:";
    pool.printStatus();

    pool.free(p5);
    std::cout << "\nAfter freeing 512 bytes:";
    pool.printStatus();
}
}  // namespace rtp_llm::threefs

int main() {
    rtp_llm::threefs::mempoolTest();
    return 0;
}