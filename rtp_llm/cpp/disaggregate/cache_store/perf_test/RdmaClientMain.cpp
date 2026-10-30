#include "rtp_llm/cpp/disaggregate/cache_store/Interface.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClient.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaConnection.h"
#include "autil/LockFreeThreadPool.h"
#include <unistd.h>
#include <mutex>
#include "kmonitor/client/KMonitorFactory.h"

using namespace rtp_llm;
using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " <ip> <port>" << std::endl;
        return -1;
    }

    std::string ip   = argv[1];
    int         port = atoi(argv[2]);
    AUTIL_ROOT_LOG_CONFIG();
    AUTIL_ROOT_LOG_SETLEVEL(INFO);

    // Init kmonitor
    std::string kmonConf = R"(
        {
            "tenant_name": "default",
            "service_name": "rtp_llm",
            "system_metrics": true,
            "sink_period": 1,
            "sink_address": "localhost:4141",
            "global_tags": {
            },
            "assigned": false
        }
    )";
    kmonitor::KMonitorFactory::Init(kmonConf);
    kmonitor::KMonitorFactory::Start();

    auto memory_util = createMemoryUtilImpl(true);
    auto rdma_client = std::make_shared<RdmaClient>(memory_util);

    if (!rdma_client || !rdma_client->init(4, 2)) {
        std::cout << "make rdma client failed" << std::endl;
        return -1;
    }

    autil::LockFreeThreadPool thread_pool(2, 100, nullptr, "RdmaClient");
    if (!thread_pool.start()) {
        std::cout << "start thread pool failed" << std::endl;
        return -2;
    }

    std::vector<std::shared_ptr<RdmaConnection>> connections;
    std::mutex                                   connections_mutex;
    auto                                         make_connection_func =
        [rdma_client, ip, port, &connections, &connections_mutex]() {
            auto connection = rdma_client->getConnection(ip, port);
            std::lock_guard<std::mutex> lock(connections_mutex);
            connections.push_back(connection);
        };

    for (int i = 0; i < 1000; i++) {
        if (thread_pool.pushTask(make_connection_func) != autil::ThreadPoolBase::ERROR_NONE) {
            std::cout << "push task to thread pool failed" << std::endl;
            continue;
        }
    }

    while (thread_pool.getItemCount() > 0) {
        sleep(1);
    }

    sleep(5);

    thread_pool.stop();
    std::cout << "all connections created" << std::endl;

    std::cout << "ready to exit";
    return 0;
}