#pragma once
#ifdef BUILD_MULTI_GPU_TCP

#include <string>
#include "src/fastertransformer/utils/nccl_utils.h"

namespace c10d {
class TCPStore;
}

namespace fastertransformer {
  
c10d::TCPStore* createTcpStore(const std::string& master_ip, 
                               const int port,  
                               const int world_size, 
                               const int rank);

void setUniqueId(ncclUniqueId* id, const std::string& store_key, 
                 c10d::TCPStore* tcp_store);
void getUniqueId(ncclUniqueId* id, const std::string& store_key, 
                 c10d::TCPStore* tcp_store);
}

#endif

