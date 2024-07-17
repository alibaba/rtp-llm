#include <gtest/gtest.h>
#include <torch/torch.h>
#include <future>

#define private public
#include "src/fastertransformer/devices/testing/TestBase.h"

using namespace std;
using namespace fastertransformer;

#define CHECK_TRUE(call)                                                                                               \
    do {                                                                                                               \
        if (!call) {                                                                                                   \
            std::stringstream ss;                                                                                      \
            ss << "Failed at " << std::string(__FILE__) << ":" << std::to_string(__LINE__) << ": " << #call << "\n";   \
            std::cerr << ss.str();                                                                                     \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define copy_tensor_to_buffer(t, buf)                                                                                  \
    {                                                                                                                  \
        auto buf_host = torchTensor2Buffer(t);                                                                         \
        device->copy({*buf, *buf_host});                                                                               \
    }

class CustomAllReduceTest: public DeviceTestBase {
public:
    void initTestDevices() override {}

    DeviceBase* initTestDevices(const size_t rank, const size_t world_size, const size_t port) {
        auto device_name = getenv("TEST_USING_DEVICE");
        CHECK_TRUE(device_name);
        auto             device_type    = getDeviceType(device_name);
        auto             device_creator = DeviceFactory::getRegistrationMap().at(device_type);
        DeviceInitParams params;
        params.device_id   = rank;
        params.tp_rank     = rank;
        params.tp_size     = world_size;
        params.master_ip   = "127.0.0.1";
        params.master_port = port;
        return device_creator(params);
    }

    void baseTest(const size_t rank, const size_t world_size, const size_t port) {
        auto device = initTestDevices(rank, world_size, port);

        // test castom all reduce
        const auto tensor = torch::arange(0, -1, -0.01, torch::kFloat32) * ((int32_t)rank + 1);
        auto       buf    = device->allocateBuffer({DataType::TYPE_FP32, {100}});
        copy_tensor_to_buffer(tensor, buf);
        device->allReduce({buf, ReduceOp::Sum});
        device->syncAndCheck();
        auto out = bufferToTensor(*buf, device);
        device->syncAndCheck();

        auto expected = torch::arange(0, -1, -0.01, torch::kFloat32)
                        * (((int32_t)world_size * ((int32_t)world_size - 1) / 2) + (int32_t)world_size);
        CHECK_TRUE(checkTensorClose(expected, out, 1e-6, 1e-6));

        fflush(stdout);
        fflush(stderr);
    }

    void executeBenchmarkRun(DeviceBase*  device,
                      const size_t rank,
                      const size_t world_size,
                      const size_t warm_iter,
                      const size_t iter_num,
                      const size_t m,
                      bool         custom_ar = true,
                      bool         log       = true) {
        float begin = 0;
        float end   = 1;
        float step  = (end - begin) / (float)(m);

        const auto tensor = torch::arange(begin, end, step, torch::kFloat32) * ((int32_t)rank + 1);
        auto       buf    = device->allocateBuffer({DataType::TYPE_FP32, {m}});
        copy_tensor_to_buffer(tensor, buf);

        for (size_t i = 0; i < warm_iter; ++i) {
            device->allReduce({buf, ReduceOp::Sum});
            device->syncAndCheck();
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < iter_num; ++i) {
            device->allReduce({buf, ReduceOp::Sum});
            device->syncAndCheck();
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        if (rank == 0 && log) {
            FT_LOG_INFO("[%s] Benchmark, world size %d, m %d, time %f us\n",
                        custom_ar ? "CUSTOM_AR" : "NCCL",
                        world_size,
                        m,
                        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }
        fflush(stdout);
        fflush(stderr);
    }

    void benchmark(const size_t rank,
                   const size_t world_size,
                   size_t port) {
        vector<unsigned long> batch_size  = {1, 8, 16, 32};
        vector<unsigned long> seq_length  = {2048, 4096};
        vector<unsigned long> hidden_size = {128, 128, 128, 4096, 5120, 8192};
        vector<unsigned long> sz_vec;
        for (auto h : hidden_size) {
            for (auto b : batch_size) {
                sz_vec.push_back(b * h);
            }
        }
        for (auto h : hidden_size) {
            for (auto s : seq_length) {
                sz_vec.push_back(s * h);
            }
        }

        unsetenv("FT_ENABLE_CUSTOM_AR");
        auto device = initTestDevices(rank, world_size, port);

        // cold run (ncclAllReduce)
        executeBenchmarkRun(device, rank, world_size, 5, 0, 100, false, false);
        for (auto m : sz_vec) {
            executeBenchmarkRun(device, rank, world_size, 0, 1, m, false);
        }

        // hot run (ncclAllReduce)
        executeBenchmarkRun(device, rank, world_size, 5, 0, 100, false, false);
        for (auto m : sz_vec) {
            executeBenchmarkRun(device, rank, world_size, 5, 100, m, false);
        }

        setenv("FT_ENABLE_CUSTOM_AR", "0", 1);
        port = getFreePort();
        FT_LOG_INFO("found free port %d\n", port);
        device = initTestDevices(rank, world_size, port);
        // cold run (custom all redcue)
        executeBenchmarkRun(device, rank, world_size, 5, 0, 100, false, false);
        for (auto m : sz_vec) {
            executeBenchmarkRun(device, rank, world_size, 0, 1, m);
        }

        // hot run (custom all redcue)
        executeBenchmarkRun(device, rank, world_size, 5, 0, 100, false, false);
        for (auto m : sz_vec) {
            executeBenchmarkRun(device, rank, world_size, 5, 100, m);
        }
    }

    void testForWorldSizeMultiProcess(const size_t world_size, bool run_benchmark = false) {

        pid_t pids[world_size];
        int   status;

        const auto port = getFreePort();
        FT_LOG_INFO("found free port %d\n", port);

        // Spawn child processes
        for (size_t i = 0; i < world_size; ++i) {
            pids[i] = fork();

            if (pids[i] == 0) {
                if (!run_benchmark) {
                    baseTest(i, world_size, port);
                } else {
                    benchmark(i, world_size, port);
                }
                _exit(EXIT_SUCCESS);
            }
        }

        bool all_children_successful = true;
        for (int i = 0; i < world_size; ++i) {
            if (waitpid(pids[i], &status, 0) == -1) {
                FT_LOG_INFO("Error waitpid");
                all_children_successful = false;
            } else if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
                FT_LOG_INFO("Child process %d exited with status %d\n", pids[i], WEXITSTATUS(status));
                all_children_successful = false;
            } else if (WIFSIGNALED(status)) {
                FT_LOG_INFO("Child process %d killed by signal %d\n", pids[i], WEXITSTATUS(status));
                all_children_successful = false;
            }
        }

        EXPECT_TRUE(all_children_successful);
    }
};

TEST_F(CustomAllReduceTest, testCustomAllReduceTest) {
    if (getenv("SKIP_DISTRIBUTED_TEST")) {
        FT_LOG_INFO("CustomAllReduce test skipped\n");
        return;
    }

    testForWorldSizeMultiProcess(2, false);
    testForWorldSizeMultiProcess(4, false);

    if (getenv("BENCHMARK_AR_TEST")) {
        FT_LOG_INFO("CustomAllReduce benchmark\n");
        testForWorldSizeMultiProcess(2, true);
        testForWorldSizeMultiProcess(4, true);
        return;
    }
}
