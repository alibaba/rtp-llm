#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/test_util/BlockBufferUtil.h"
#include <unistd.h>
#include "kmonitor/client/KMonitorFactory.h"

using namespace rtp_llm;
using namespace std;

volatile bool gStopFlag = false;
void          handleSignal(int sig) {
    std::cout << "receive server stopped sig: " << sig << std::endl;
    gStopFlag = true;
}

void registerSignalHandler() {
    signal(SIGINT, handleSignal);
    signal(SIGQUIT, handleSignal);
    signal(SIGTERM, handleSignal);
    signal(SIGUSR1, handleSignal);
    signal(SIGUSR2, handleSignal);
}

void updateStoreResult(bool                                     ok,
                       CacheStoreErrorCode                      ec,
                       const std::shared_ptr<NormalCacheStore>& cache_store_client,
                       int                                      request_id,
                       int                                      layer_id,
                       int                                      layer_num,
                       std::shared_ptr<std::atomic_int>         load_layer_cnt) {
    if (ok) {
        ++(*load_layer_cnt);
    }
    if (load_layer_cnt->load() == layer_num) {
        std::cout << "store local success request_id" << request_id << std::endl;
        usleep(3000 * 1000);
        cache_store_client->markRequestEnd("test-request" + std::to_string(request_id));
        std::cout << "markend requestid" << request_id << std::endl;
    }
}

int main(int argc, char** argv) {
    registerSignalHandler();

    AUTIL_ROOT_LOG_CONFIG();
    AUTIL_ROOT_LOG_SETLEVEL(INFO);

    // Init kmonitor
    std::string kmonConf = R"(
        {
            "tenant_name": "default",
            "service_name": "arpc",
            "system_metrics": true,
            "sink_period": 1,
            "sink_address": "localhost:4141",
            "global_tags": {
                "app":"cache_store_client"
            },
            "assigned": false
        }
    )";
    kmonitor::KMonitorFactory::Init(kmonConf);
    kmonitor::KMonitorFactory::Start();
    DeviceResourceConfig device_resource_config;
    auto                 device_util                   = std::make_shared<DeviceUtil>(device_resource_config);
    CacheStoreInitParams client_params;
    client_params.listen_port = 12348;
    client_params.rdma_mode   = false;
    client_params.device_id   = 0;

    auto cache_store_client = NormalCacheStore::createNormalCacheStore(client_params);
    if (!cache_store_client) {
        std::cout << "make cache store client failed" << std::endl;
        return -1;
    }

    auto block_buffer_util = std::make_shared<BlockBufferUtil>(cache_store_client->getMemoryUtil(), device_util);

    int      layer_num  = 4;
    int      block_num  = 125;
    uint32_t block_size = 32 * 1024;

    std::atomic_int request_id = 0;
    while (!gStopFlag) {

        while (!gStopFlag
               && cache_store_client->getRequestBlockBufferStore()->getRequestBlockBuffer("test-request"
                                                                                          + std::to_string(request_id))
                      == nullptr) {
            usleep(1000);
        }
        auto store_caches = block_buffer_util->makeRequestBlockBufferVec(
            "test-request" + std::to_string(request_id), layer_num, block_num, block_size);
        if (store_caches.empty()) {
            std::cout << "make store cache failed" << std::endl;
            return -2;
        }
        auto load_layer_cnt = std::make_shared<std::atomic_int>(0);
        for (size_t layer_id = 0; layer_id < store_caches.size(); layer_id++) {
            CacheStoreStoreDoneCallback layer_callback =
                [cache_store_client, call_back_request_id = request_id.load(), layer_id, layer_num, load_layer_cnt](
                    bool ok, CacheStoreErrorCode ec) {
                    // update store layer count and mark request end
                    updateStoreResult(
                        ok, ec, cache_store_client, call_back_request_id, layer_id, layer_num, load_layer_cnt);
                };
            cache_store_client->store(store_caches[layer_id], layer_callback);
        }
        ++request_id;
    }

    return 0;
}
