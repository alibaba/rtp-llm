
#include <csignal>
#include <iostream>
#include <random>
#include <vector>

#include "rtp_llm/cpp/cache/DistStorage3FS.h"

using namespace rtp_llm;
using namespace rtp_llm::threefs;

std::atomic<bool> g_stop_flag{false};

void handleSignal(int sig) {
    printf("recvd stopped signal: %d\n", sig);
    g_stop_flag.store(true);
}

void registerSignalHandler() {
    signal(SIGINT, handleSignal);
    signal(SIGQUIT, handleSignal);
    signal(SIGTERM, handleSignal);
    signal(SIGUSR1, handleSignal);
    signal(SIGUSR2, handleSignal);
}

void startTest() {

    DistStorage3FSInitParams init_params;
    init_params.root_dir = "test/";

    auto storage_3fs = std::make_shared<DistStorage3FS>(nullptr);
    if (!storage_3fs->init(init_params)) {
        RTP_LLM_LOG_WARNING("storage 3fs init failed");
        return;
    }

    DistStorage::Item item;
    item.type = DistStorage::StorageType::ST_3FS;
    item.key  = "/3fs/stage/3fs/test/kv_test";

    const int buffer_len    = 1 * 1024 * 1024;
    auto      buffer        = malloc(buffer_len);
    auto      shared_buffer = std::shared_ptr<void>(buffer, [](void* ptr) { free(ptr); });
    int64_t   offset        = 0;
    for (int i = 0; i < 10; ++i) {
        DistStorage::Iov iov;
        iov.data    = std::shared_ptr<void>(buffer, [](void* ptr) {});
        iov.len     = buffer_len;
        iov.offset  = offset;
        iov.gpu_mem = false;
        item.iovs.push_back(iov);
    }

    if (!storage_3fs->put(item)) {
        RTP_LLM_LOG_WARNING("storage 3fs put failed");
        return;
    }
    RTP_LLM_LOG_INFO("storage 3fs put success");
}

int main(int argc, char* argv[]) {
    registerSignalHandler();

    AUTIL_ROOT_LOG_CONFIG();
    AUTIL_ROOT_LOG_SETLEVEL(INFO);

    startTest();
    RTP_LLM_LOG_INFO("---------------------- Test Finished ----------------------");
    return 0;
}