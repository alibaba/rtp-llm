#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/test_util/BlockBufferUtil.h"
#include <unistd.h>
#include "kmonitor/client/KMonitorFactory.h"
#include "autil/NetUtil.h"

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
                "app":"cache_store_server"
            },
            "assigned": false
        }
    )";
    kmonitor::KMonitorFactory::Init(kmonConf);
    kmonitor::KMonitorFactory::Start();
    DeviceResourceConfig device_resource_config;
    auto device_util                                   = std::make_shared<DeviceUtil>(device_resource_config);

    CacheStoreInitParams server_params;
    server_params.listen_port = 54321;
    server_params.rdma_mode   = false;
    server_params.device_id   = 0;

    auto cache_store_server = NormalCacheStore::createNormalCacheStore(server_params);
    if (!cache_store_server) {
        std::cout << "make cache store server failed" << std::endl;
        return -1;
    }

    auto block_buffer_util = std::make_shared<BlockBufferUtil>(cache_store_server->getMemoryUtil(), device_util);

    int      layer_num  = 4;
    int      block_num  = 125;
    uint32_t block_size = 32 * 1024;

    std::atomic_int request_id = 0;

    while (!gStopFlag) {
        auto load_caches = block_buffer_util->makeRequestBlockBufferVec(
            "test-request" + std::to_string(request_id), layer_num, block_num, block_size);
        // the paired CacheStoreClient harness serves its store on port 12348
        auto load_context = cache_store_server->loadBuffers(
            load_caches, autil::NetUtil::getBindIp(), 12348, 0, 5000, nullptr, 1, 0);
        load_context->waitDone();
        if (load_context->success()) {
            std::cout << "load success request_id" << request_id.load() << std::endl;
            ++request_id;
        } else {
            std::cout << "load failed request_id" << request_id.load() << std::endl;
            sleep(1);
        }
    }
    return 0;
}
