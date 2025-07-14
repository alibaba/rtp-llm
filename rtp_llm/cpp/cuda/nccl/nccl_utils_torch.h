#pragma once

#include "nccl_utils.h"
#include <string>

namespace c10d {
class TCPStore;
}

namespace rtp_llm {

c10d::TCPStore* createTcpStore(const std::string& master_ip, const int port, const int world_size, const int rank);

void setUniqueId(ncclUniqueId* id, const std::string& store_key, c10d::TCPStore* tcp_store);
void getUniqueId(ncclUniqueId* id, const std::string& store_key, c10d::TCPStore* tcp_store);

void all2all_single_equal_split(
    void* sendbuff, void* recvbuff, size_t total_size, ncclComm_t comm, cudaStream_t stream);

void all2all_single_unequal_split(void*          sendbuff,
                                  const size_t*  sendcounts,
                                  const size_t*  senddispls,
                                  void*          recvbuff,
                                  const size_t*  recvcounts,
                                  const size_t*  recvdispls,
                                  size_t         size,
                                  ncclDataType_t type,
                                  ncclComm_t     comm,
                                  cudaStream_t   stream);

}  // namespace rtp_llm
