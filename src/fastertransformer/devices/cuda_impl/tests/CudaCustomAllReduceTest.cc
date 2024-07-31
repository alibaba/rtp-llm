#include <cstdio>
#include <gtest/gtest.h>
#include <spawn.h>

#define private public
#include "src/fastertransformer/devices/testing/TestBase.h"

using namespace std;
using namespace fastertransformer;

extern char** environ;

class CudaCustomAllReduceTest: public DeviceTestBase {
    void initTestDevices() override {}

public:
    void testForWorldSizeMultiProcess(const size_t world_size, bool run_benchmark = false) {

        pid_t pids[world_size];
        int   status;

        const auto port = getFreePort();
        FT_LOG_INFO("found free port %d", port);

        const auto port2 = getFreePort();
        FT_LOG_INFO("found free port2 %d", port2);

        posix_spawn_file_actions_t action;
        posix_spawnattr_t          attr;

        posix_spawnattr_init(&attr);
        posix_spawn_file_actions_init(&action);

        const std::string custom_ar_test_executable_path = "../../../../cuda_impl/tests/custom_ar_test";

        // Spawn child processes
        for (size_t i = 0; i < world_size; ++i) {
            char benchmark[32], i_str[32], world_size_str[32], port_str[32], port2_str[32];
            snprintf(benchmark, sizeof(benchmark), "%d", run_benchmark);
            snprintf(i_str, sizeof(i_str), "%zu", i);
            snprintf(world_size_str, sizeof(world_size_str), "%zu", world_size);
            snprintf(port_str, sizeof(port_str), "%d", port);
            snprintf(port2_str, sizeof(port2_str), "%d", port2);

            char* argv[] = {(char*)custom_ar_test_executable_path.c_str(),
                            benchmark,
                            i_str,
                            world_size_str,
                            port_str,
                            port2_str,
                            nullptr};

            std::string ld_library_path = "LD_LIBRARY_PATH=";
            ld_library_path += std::getenv("LD_LIBRARY_PATH");
            std::vector<std::string> custom_env_vars = {"TEST_USING_DEVICE=CUDA", ld_library_path};

            std::vector<char*> envp;
            for (const std::string& env_var : custom_env_vars) {
                envp.push_back(const_cast<char*>(env_var.c_str()));
            }
            envp.push_back(nullptr);

            int ret =
                posix_spawn(&pids[i], (char*)custom_ar_test_executable_path.c_str(), &action, &attr, argv, envp.data());
            if (ret != 0) {
                FT_LOG_INFO("posix_spawn failed: %s", strerror(ret));
                exit(EXIT_FAILURE);
            }
        }

        posix_spawn_file_actions_destroy(&action);
        posix_spawnattr_destroy(&attr);

        bool all_children_successful = true;

        for (size_t i = 0; i < world_size; ++i) {
            if (waitpid(pids[i], &status, 0) == -1) {
                FT_LOG_INFO("Error waitpid");
                all_children_successful = false;
            } else if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
                FT_LOG_INFO("Child process %d exited with status %d", pids[i], WEXITSTATUS(status));
                all_children_successful = false;
            } else if (WIFSIGNALED(status)) {
                FT_LOG_INFO("Child process %d killed by signal %d", pids[i], WTERMSIG(status));
                all_children_successful = false;
            }
        }

        EXPECT_TRUE(all_children_successful);
    }
};

TEST_F(CudaCustomAllReduceTest, base) {
    if (getenv("SKIP_DISTRIBUTED_TEST")) {
        FT_LOG_INFO("CustomAllReduce test skipped");
        return;
    }

    testForWorldSizeMultiProcess(2, false);
    testForWorldSizeMultiProcess(4, false);
}

TEST_F(CudaCustomAllReduceTest, benchmark) {
    if (getenv("BENCHMARK_AR_TEST")) {
        FT_LOG_INFO("CustomAllReduce benchmark");
        testForWorldSizeMultiProcess(2, true);
        testForWorldSizeMultiProcess(4, true);
        return;
    }
}
