#include "nccl_utils_torch.h"
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

namespace fastertransformer {

#ifdef BUILD_MULTI_GPU_TCP
c10d::TCPStore* createTcpStore(const std::string& master_ip, const int port, const int world_size, const int rank) {
    c10d::TCPStoreOptions options;
    options.port          = port;
    options.isServer      = rank == 0;
    options.waitWorkers   = true;
    c10d::TCPStore* store = new c10d::TCPStore(master_ip, options);
    return store;
}

void setUniqueId(ncclUniqueId* id, const std::string& store_key, c10d::TCPStore* tcp_store) {
    try {
        auto vec =
            std::vector<uint8_t>(reinterpret_cast<uint8_t*>(id), reinterpret_cast<uint8_t*>(id) + NCCL_UNIQUE_ID_BYTES);
        tcp_store->set(store_key, vec);
    } catch (const std::exception& e) { FT_CHECK_WITH_INFO(false, "failed to set unique id"); }
}

void getUniqueId(ncclUniqueId* id, const std::string& store_key, c10d::TCPStore* tcp_store) {
    try {
        auto vec = tcp_store->get(store_key);
        TORCH_CHECK(vec.size() == NCCL_UNIQUE_ID_BYTES);
        std::memcpy(id, vec.data(), vec.size());
    } catch (const std::exception& e) { FT_CHECK_WITH_INFO(false, "failed to get unique id"); }
}
#endif

}  // namespace fastertransformer
