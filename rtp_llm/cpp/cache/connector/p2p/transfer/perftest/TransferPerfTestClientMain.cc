#include <iostream>
#include <getopt.h>

#include "autil/Log.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/perftest/TransferPerfTestClient.h"

namespace rtp_llm {

const std::string PERFTEST_DEFAULT_LOG_CONF = R"conf(
alog.rootLogger=INFO, perftestAppender
alog.max_msg_len=4096
alog.appender.perftestAppender=ConsoleAppender
alog.appender.perftestAppender.flush=true
alog.appender.perftestAppender.layout=PatternLayout
alog.appender.perftestAppender.layout.LogPattern=[%%d] [%%l] [%%t,%%F -- %%f():%%n] [%%m]
alog.logger.arpc=WARN
alog.logger.kmonitor=WARN
)conf";

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n"
              << "Options:\n"
              << "  --server_ip IP              Server IP address (default: 127.0.0.1)\n"
              << "  --server_port PORT          Server port (default: 8888)\n"
              << "  --block_count COUNT         Number of blocks per transfer (default: 10)\n"
              << "  --block_size SIZE           Size of each block in bytes (default: 1048576)\n"
              << "  --transfer_count COUNT      Number of transfers to perform (default: 100)\n"
              << "  --tcp_io_threads COUNT      TCP IO thread count (default: 4)\n"
              << "  --timeout_ms MS             Timeout in milliseconds (default: 10000)\n"
              << "  --use_rdma                  Use RDMA for transfer (default: false)\n"
              << "  --help                      Show this help message\n";
}

PerfTestClientConfig parseArgs(int argc, char* argv[]) {
    PerfTestClientConfig config;

    static struct option long_options[] = {{"server_ip", required_argument, 0, 'a'},
                                           {"server_port", required_argument, 0, 'p'},
                                           {"block_count", required_argument, 0, 'b'},
                                           {"block_size", required_argument, 0, 's'},
                                           {"transfer_count", required_argument, 0, 't'},
                                           {"tcp_io_threads", required_argument, 0, 'i'},
                                           {"timeout_ms", required_argument, 0, 'o'},
                                           {"use_rdma", no_argument, 0, 'r'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "a:p:b:s:t:i:o:rh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'a':
                config.server_ip = optarg;
                break;
            case 'p':
                config.server_port = std::stoul(optarg);
                break;
            case 'b':
                config.block_count = std::stoi(optarg);
                break;
            case 's':
                config.block_size = std::stoull(optarg);
                break;
            case 't':
                config.transfer_count = std::stoi(optarg);
                break;
            case 'i':
                config.tcp_io_thread_count = std::stoi(optarg);
                break;
            case 'o':
                config.timeout_ms = std::stoi(optarg);
                break;
            case 'r':
                config.use_rdma = true;
                break;
            case 'h':
            default:
                printUsage(argv[0]);
                exit(opt == 'h' ? 0 : 1);
        }
    }

    return config;
}

}  // namespace rtp_llm

int main(int argc, char* argv[]) {
    // 初始化 kmonitor
    rtp_llm::initKmonitorFactory();

    // 初始化日志
    AUTIL_LOG_CONFIG_FROM_STRING(rtp_llm::PERFTEST_DEFAULT_LOG_CONF.c_str());

    // 解析命令行参数
    auto config = rtp_llm::parseArgs(argc, argv);

    // 创建并运行 Client
    auto client = std::make_shared<rtp_llm::TransferPerfTestClient>(config);

    if (!client->init()) {
        std::cerr << "Failed to initialize TransferPerfTestClient" << std::endl;
        rtp_llm::stopKmonitorFactory();
        return -1;
    }

    auto ret = client->run();

    client.reset();

    // 停止 kmonitor
    rtp_llm::stopKmonitorFactory();

    return ret;
}
