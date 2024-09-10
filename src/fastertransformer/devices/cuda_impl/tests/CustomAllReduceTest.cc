#include <cstddef>
#include <torch/torch.h>
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"

#define private public
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/CudaDevice.h"

using namespace std;
using namespace fastertransformer;

// Note: used for catching error code for multiprocess test, do not remove
#define CHECK_TRUE(call)                                                                                               \
    do {                                                                                                               \
        if (!call) {                                                                                                   \
            std::stringstream ss;                                                                                      \
            ss << "Failed at " << std::string(__FILE__) << ":" << std::to_string(__LINE__) << ": " << #call << "\n";   \
            std::cerr << ss.str();                                                                                     \
            fflush(stdout);                                                                                            \
            fflush(stderr);                                                                                            \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define copy_tensor_to_buffer(t, buf)                                                                                  \
    {                                                                                                                  \
        auto buf_host = torchTensor2Buffer(t);                                                                         \
        device->copy({*buf, *buf_host});                                                                               \
    }

torch::Tensor bufferToTensor(const Buffer& buffer, DeviceBase* device) {
    auto host_buffer = device->allocateBuffer({buffer.type(), buffer.shape(), AllocationType::HOST});
    device->copy({*host_buffer, buffer});
    device->syncAndCheck();

    return torch::from_blob(
               host_buffer->data(),
               bufferShapeToTorchShape(buffer),
               c10::TensorOptions().device(torch::Device(torch::kCPU)).dtype(dataTypeToTorchType(buffer.type())))
        .clone();
}

// Note: used for catching error code for multiprocess test, do not remove
bool checkTensorClose(const torch::Tensor& a, const torch::Tensor& b, double rtol = 0, double atol = 0) {
    auto a_cmp = a;
    auto b_cmp = b;
    if (a.is_floating_point() != b.is_floating_point()) {
        return false;
    }

    if (a_cmp.dtype() != b_cmp.dtype()) {
        auto cmp_type = (a_cmp.dtype().itemsize() > b_cmp.dtype().itemsize()) ? a_cmp.dtype() : b_cmp.dtype();
        a_cmp         = a_cmp.to(cmp_type);
        b_cmp         = b_cmp.to(cmp_type);
    }
    a_cmp = a_cmp.squeeze();
    b_cmp = b_cmp.squeeze();

    const auto close = torch::allclose(a_cmp, b_cmp, rtol, atol);
    if (!close) {
        std::cout << "assert tensor close failed!" << std::endl;
        std::cout << "rtol: " << rtol << std::endl;
        std::cout << "atol: " << atol << std::endl;
        std::cout << "a: " << a << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << "abs diff: " << torch::abs(a_cmp - b_cmp) << std::endl;
        std::cout << "rel diff: " << torch::abs(a_cmp - b_cmp) / torch::abs(a_cmp) << std::endl;
        return false;
    }
    return true;
}

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

void baseTest(const size_t rank, const size_t world_size, const size_t port, const size_t m) {
    auto device = initTestDevices(rank, world_size, port);

    // test castom all reduce
    const float begin = 0.0;
    const float end   = 1.0;
    const float step  = (end - begin) / m;

    const auto tensor = torch::arange(begin, end, step, torch::kFloat32) * ((int32_t)rank + 1);
    auto       buf    = device->allocateBuffer({DataType::TYPE_FP32, {static_cast<unsigned long>(tensor.size(0))}});
    buf               = device->prepareAllReduce({std::move(buf), ReduceOp::Sum}).buffer;
    copy_tensor_to_buffer(tensor, buf);
    buf = device->allReduce({buf, ReduceOp::Sum}).buffer;
    device->syncAndCheck();
    auto out = bufferToTensor(*buf, device);
    device->syncAndCheck();

    auto expected = torch::arange(begin, end, step, torch::kFloat32)
                    * (((int32_t)world_size * ((int32_t)world_size - 1) / 2) + (int32_t)world_size);
    CHECK_TRUE(checkTensorClose(expected, out, 1e-6, 1e-6));
}

void executeBenchmarkRun(DeviceBase*  device,
                         const size_t rank,
                         const size_t world_size,
                         const size_t warm_iter,
                         const size_t iter_num,
                         const size_t m,
                         bool         custom_ar = true,
                         bool         log       = true) {

    const auto tensor = torch::ones({int(m)}, torch::kFloat32) * 0.01 * ((int32_t)rank + 1);
    auto       buf    = device->allocateBuffer({DataType::TYPE_FP32, {m}});
    device->syncAndCheck();

    for (size_t i = 0; i < warm_iter; ++i) {
        buf = device->prepareAllReduce({std::move(buf), ReduceOp::Sum}).buffer;
        copy_tensor_to_buffer(tensor, buf);
        buf = device->allReduce({buf, ReduceOp::Sum}).buffer;
    }
    device->syncAndCheck();

    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iter_num; ++i) {
        buf = device->prepareAllReduce({std::move(buf), ReduceOp::Sum}).buffer;
        copy_tensor_to_buffer(tensor, buf);
        buf = device->allReduce({buf, ReduceOp::Sum}).buffer;
    }
    device->syncAndCheck();
    auto end_time = std::chrono::high_resolution_clock::now();
    if (rank == 0 && log) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        FT_LOG_INFO("[%s] Benchmark, world size %d, data size %d, time %d us",
                    custom_ar ? "CUSTOM_AR" : "NCCL",
                    world_size,
                    m,
                    duration.count() / iter_num);
    }

    fflush(stdout);
    fflush(stderr);
}

void benchmark(const size_t rank, const size_t world_size, size_t port, size_t port2) {
    vector<unsigned long> batch_size  = {1, 8, 16, 32};
    vector<unsigned long> seq_length  = {2048, 4096};
    vector<unsigned long> hidden_size = {4096, 5120, 8192};
    vector<unsigned long> sz_vec;
    for (auto h : hidden_size) {
        for (auto b : batch_size) {
            sz_vec.push_back(b * h / world_size);
        }
    }
    for (auto h : hidden_size) {
        for (auto s : seq_length) {
            sz_vec.push_back(s * h / world_size);
        }
    }

    setenv("FT_DISABLE_CUSTOM_AR", "1", 1);

    auto device = initTestDevices(rank, world_size, port);

    // cold run (ncclAllReduce)
    FT_LOG_INFO("[NCCL] Start cold run");
    executeBenchmarkRun(device, rank, world_size, 5, 0, 100, false, false);
    for (auto m : sz_vec) {
        executeBenchmarkRun(device, rank, world_size, 0, 1, m, false);
    }

    // hot run (ncclAllReduce)
    FT_LOG_INFO("[NCCL] Start hot run");
    executeBenchmarkRun(device, rank, world_size, 5, 0, 100, false, false);
    for (auto m : sz_vec) {
        executeBenchmarkRun(device, rank, world_size, 5, 100, m, false);
    }

    unsetenv("FT_DISABLE_CUSTOM_AR");
    device = initTestDevices(rank, world_size, port2);
    // cold run (custom all redcue)
    FT_LOG_INFO("[Custom AR] Start cold run");
    executeBenchmarkRun(device, rank, world_size, 5, 0, 100, false, false);
    for (auto m : sz_vec) {
        executeBenchmarkRun(device, rank, world_size, 0, 1, m);
    }

    // hot run (custom all redcue)
    FT_LOG_INFO("[Custom AR] Start hot run");
    executeBenchmarkRun(device, rank, world_size, 5, 0, 100, false, false);
    for (auto m : sz_vec) {
        executeBenchmarkRun(device, rank, world_size, 5, 100, m);
    }
}

void parse_arguments(
    int argc, char* argv[], int* run_benchmark, size_t* rank, size_t* world_size, int* port, int* port2) {
    if (argc != 6) {
        FT_LOG_INFO("argc %d\n", argc);
        FT_LOG_INFO("Usage: %s <run_benchmark> <rank> <world_size> <port> <port2>", argv[0]);
        exit(EXIT_FAILURE);
    }

    *run_benchmark = atoi(argv[1]);
    *rank          = (size_t)strtoul(argv[2], NULL, 10);
    *world_size    = (size_t)strtoul(argv[3], NULL, 10);
    *port          = atoi(argv[4]);
    *port2         = atoi(argv[5]);
}

int main(int argc, char* argv[]) {
    int    run_benchmark = 0;
    size_t rank          = 0;
    size_t world_size    = 0;
    int    port          = 0;
    int    port2         = 0;

    parse_arguments(argc, argv, &run_benchmark, &rank, &world_size, &port, &port2);

    FT_LOG_INFO("run_benchmark: %d", run_benchmark);
    FT_LOG_INFO("world_size: %zu", world_size);
    FT_LOG_INFO("rank: %zu", rank);
    FT_LOG_INFO("port: %d", port);
    FT_LOG_INFO("port2: %d", port2);

    fflush(stdout);

    if (run_benchmark == 1) {
        benchmark(rank, world_size, port, port2);
    } else {
        baseTest(rank, world_size, port, 128);
        baseTest(rank, world_size, port2, 1048576);
    }

    return 0;
}
