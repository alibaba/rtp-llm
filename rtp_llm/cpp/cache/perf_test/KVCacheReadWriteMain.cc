#include <random>

#include "rtp_llm/cpp/cache/perf_test/KVCacheReadPerfTest.h"
#include "rtp_llm/cpp/cache/perf_test/KVCacheWritePerfTest.h"

using namespace rtp_llm;

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

int main(int argc, char* argv[]) {
    registerSignalHandler();

    AUTIL_ROOT_LOG_CONFIG();
    AUTIL_ROOT_LOG_SETLEVEL(INFO);

    KVCacheOptionBase option_base;
    if (!option_base.parseOptions(argc, argv)) {
        RTP_LLM_LOG_ERROR("parse base options failed");
        return -1;
    }

    setenv("DEVICE_RESERVE_MEMORY_BYTES", "1073741824", 0);
    setenv("ENABLE_3FS", "0", 0);
    setenv("BIZ_NAME", "test_biz", 0);
    setenv("CHECKPOINT_PATH", "test_ckpt_path", 0);
    setenv("LORA_CKPT_PATH", "test_lora_ckpt_path", 0);

    if (option_base.read) {
        KVCacheReadOption read_option;
        if (!read_option.parseOptions(argc, argv)) {
            RTP_LLM_LOG_ERROR("parse read options failed");
            return -1;
        }
        RTP_LLM_LOG_INFO("read option: [%s]", read_option.toString().c_str());
        KVCacheReadPerfTest read_test;
        read_test.startTest(read_option);
    } else if (option_base.write) {
        KVCacheWriteOption write_option;
        if (!write_option.parseOptions(argc, argv)) {
            RTP_LLM_LOG_ERROR("parse write options failed");
            return -1;
        }
        RTP_LLM_LOG_INFO("write option: [%s]", write_option.toString().c_str());
        KVCacheWritePerfTest write_test;
        write_test.startTest(write_option);
    } else {
        RTP_LLM_LOG_ERROR("read or write option must be set!");
        return -1;
    }

    RTP_LLM_LOG_INFO("---------------------- Test Finished ----------------------");
    return 0;
}