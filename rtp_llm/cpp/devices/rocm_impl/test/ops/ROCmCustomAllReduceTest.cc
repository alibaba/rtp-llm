#include <cstdio>
#include <gtest/gtest.h>
#include <spawn.h>

#define private public
#include "rtp_llm/cpp/devices/testing/TestBase.h"

using namespace std;
using namespace rtp_llm;

extern char** environ;

class ROCmCustomAllReduceTest: public DeviceTestBase {
    void initTestDevices() override {}

public:
    void testForWorldSizeMultiProcess(const size_t world_size, bool run_benchmark = false) {

        pid_t pids[world_size];
        int   status;

        const auto port = getFreePort();
        RTP_LLM_LOG_INFO("found free port %d", port);

        posix_spawn_file_actions_t action;
        posix_spawnattr_t          attr;

        posix_spawnattr_init(&attr);
        posix_spawn_file_actions_init(&action);

        const std::string custom_ar_test_executable_path = "../../../../rocm_impl/test/custom_ar_test";

        // Spawn child processes
        for (size_t i = 0; i < world_size; ++i) {
            char benchmark[32], i_str[32], world_size_str[32], port_str[32];
            snprintf(benchmark, sizeof(benchmark), "%d", run_benchmark);
            snprintf(i_str, sizeof(i_str), "%zu", i);
            snprintf(world_size_str, sizeof(world_size_str), "%zu", world_size);
            snprintf(port_str, sizeof(port_str), "%ld", port);

            char* argv[] = {
                (char*)custom_ar_test_executable_path.c_str(), benchmark, i_str, world_size_str, port_str, nullptr};

            std::string ld_library_path = "LD_LIBRARY_PATH=";
            ld_library_path += std::getenv("LD_LIBRARY_PATH");
            std::vector<std::string> custom_env_vars = {"TEST_USING_DEVICE=ROCM", ld_library_path};

            if (world_size == 2) {
                custom_env_vars.push_back("CUDA_VISIBLE_DEVICES=0,1");
            } else if (world_size == 4) {
                custom_env_vars.push_back("CUDA_VISIBLE_DEVICES=0,1,2,3");
            } else if (world_size == 8) {
                custom_env_vars.push_back("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7");
            }
            std::vector<char*> envp;
            for (const std::string& env_var : custom_env_vars) {
                envp.push_back(const_cast<char*>(env_var.c_str()));
            }
            envp.push_back(nullptr);

            int ret =
                posix_spawn(&pids[i], (char*)custom_ar_test_executable_path.c_str(), &action, &attr, argv, envp.data());
            if (ret != 0) {
                RTP_LLM_LOG_INFO("posix_spawn failed: %s", strerror(ret));
                exit(EXIT_FAILURE);
            }
        }

        posix_spawn_file_actions_destroy(&action);
        posix_spawnattr_destroy(&attr);

        bool all_children_successful = true;

        for (size_t i = 0; i < world_size; ++i) {
            if (waitpid(pids[i], &status, 0) == -1) {
                RTP_LLM_LOG_INFO("Error waitpid");
                all_children_successful = false;
            } else if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
                RTP_LLM_LOG_INFO("Child process %d exited with status %d", pids[i], WEXITSTATUS(status));
                all_children_successful = false;
            } else if (WIFSIGNALED(status)) {
                RTP_LLM_LOG_INFO("Child process %d killed by signal %d", pids[i], WTERMSIG(status));
                all_children_successful = false;
            }
        }

        EXPECT_TRUE(all_children_successful);
    }
};

TEST_F(ROCmCustomAllReduceTest, base) {
    if (getenv("SKIP_DISTRIBUTED_TEST")) {
        RTP_LLM_LOG_INFO("CustomAllReduce test skipped");
        return;
    }

    testForWorldSizeMultiProcess(2, false);
    testForWorldSizeMultiProcess(4, false);
    testForWorldSizeMultiProcess(8, false);
}

TEST_F(ROCmCustomAllReduceTest, benchmark) {
    if (getenv("BENCHMARK_AR_TEST")) {
        RTP_LLM_LOG_INFO("CustomAllReduce benchmark");
        testForWorldSizeMultiProcess(4, true);
        testForWorldSizeMultiProcess(2, true);
        return;
    }
}
