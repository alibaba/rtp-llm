#include "nccl_utils_torch.h"
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

namespace rtp_llm {

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
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("failed to set unique id, exception: %s", e.what());
    }
}

void getUniqueId(ncclUniqueId* id, const std::string& store_key, c10d::TCPStore* tcp_store) {
    try {
        auto vec = tcp_store->get(store_key);
        TORCH_CHECK(vec.size() == NCCL_UNIQUE_ID_BYTES);
        std::memcpy(id, vec.data(), vec.size());
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("failed to get unique id, exception: %s", e.what());
    }
}

void all2all_single_equal_split(
    void* sendbuff, void* recvbuff, size_t total_size, ncclComm_t comm, cudaStream_t stream) {
    ncclDataType_t type     = ncclUint8;
    int            numranks = 0;
    NCCLCHECK(ncclCommCount(comm, &numranks));

    size_t rankdiff = total_size / numranks;
    NCCLCHECK(ncclGroupStart());
    for (int peer = 0; peer < numranks; ++peer) {
        NCCLCHECK(ncclSend((char*)sendbuff + peer * rankdiff, rankdiff, type, peer, comm, stream));
        NCCLCHECK(ncclRecv((char*)recvbuff + peer * rankdiff, rankdiff, type, peer, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
}

void all2all_single_unequal_split(void*          sendbuff,
                                  const size_t*  sendcounts,
                                  const size_t*  senddispls,
                                  void*          recvbuff,
                                  const size_t*  recvcounts,
                                  const size_t*  recvdispls,
                                  size_t         size,
                                  ncclDataType_t type,
                                  ncclComm_t     comm,
                                  cudaStream_t   stream) {
    int numranks = 0;
    NCCLCHECK(ncclCommCount(comm, &numranks));
    NCCLCHECK(ncclGroupStart());
    for (int peer = 0; peer < numranks; ++peer) {
        NCCLCHECK(ncclSend(((char*)sendbuff) + senddispls[peer] * size, sendcounts[peer], type, peer, comm, stream));
        NCCLCHECK(ncclRecv(((char*)recvbuff) + recvdispls[peer] * size, recvcounts[peer], type, peer, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
}

}  // namespace rtp_llm
