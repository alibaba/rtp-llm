#include <cstddef>
#include <torch/torch.h>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

#define private public
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
using namespace std;
using namespace rtp_llm;

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
    params.device_id      = rank;
    params.tp_rank        = rank;
    params.tp_size        = world_size;
    params.master_ip      = "127.0.0.1";
    params.tp_master_port = port;
    return device_creator.create(params);
}

void baseTest(const size_t rank, const size_t world_size, const size_t port, const size_t m) {
    auto device = initTestDevices(rank, world_size, port);

    for (size_t i = 1; i < 4096; i++) {
        // test castom all reduce
        const float begin = 0.0;
        const float end   = 1.0;
        const float step  = (end - begin) / (i * m);
        RTP_LLM_LOG_INFO("size: %d", i * m);

        const auto tensor = torch::arange(begin, end, step, torch::kFloat16) * ((int32_t)rank + 1);
        auto       buf    = device->allocateBuffer({DataType::TYPE_FP16, {static_cast<unsigned long>(tensor.size(0))}});
        buf               = device->prepareAllReduce({std::move(buf), ReduceOp::Sum}).buffer;
        copy_tensor_to_buffer(tensor, buf);
        buf = device->allReduce({buf, ReduceOp::Sum}).buffer;
        device->syncAndCheck();
        auto out = bufferToTensor(*buf, device);
        device->syncAndCheck();

        auto expected = torch::arange(begin, end, step, torch::kFloat16)
                        * (((int32_t)world_size * ((int32_t)world_size - 1) / 2) + (int32_t)world_size);
        CHECK_TRUE(checkTensorClose(expected, out, 1e-3, 1e-3));
    }

    for (size_t i = 1; i < 4096; i++) {
        // test castom all reduce
        const float begin = 0.0;
        const float end   = 1.0;
        const float step  = (end - begin) / (i * m);
        RTP_LLM_LOG_INFO("size: %d", i * m);

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
}

size_t executeBenchmarkRun(DeviceBase*  device,
                           const size_t rank,
                           const size_t world_size,
                           const size_t warm_iter,
                           const size_t iter_num,
                           const size_t m,
                           bool         custom_ar = true,
                           bool         log       = true) {

    const auto tensor = torch::ones({int(m)}, torch::kFloat16) * 0.01 * ((int32_t)rank + 1);
    auto       buf    = device->allocateBuffer({DataType::TYPE_FP16, {m}});
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

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    return duration.count() / max((size_t)1, iter_num);
}

void benchmark(const size_t rank, const size_t world_size, size_t port) {
    vector<unsigned long> seq_length;
    vector<unsigned long> hidden_size = {768, 896, 1024, 1536, 2048, 3584, 5120, 8192};
    vector<unsigned long> sz_vec;
    for (size_t i = 1; i < 4096; i++) {
        seq_length.push_back(i);
    }

    vector<size_t> custom_ar_times;
    vector<size_t> nccl_times;
    size_t         k     = 0;
    size_t         port2 = port - 10000;

    for (auto h : hidden_size) {

        vector<size_t>        part_custom_ar_times;
        vector<size_t>        part_nccl_times;
        vector<unsigned long> part_sz_vec;

        for (auto s : seq_length) {
            sz_vec.push_back(s * h);
            part_sz_vec.push_back(s * h);
        }

        auto device                                                   = initTestDevices(rank, world_size, port2 + k);
        device->initParamsRef().hw_kernel_config.ft_disable_custom_ar = false;
        RTP_LLM_LOG_INFO("[Custom AR] Start hot run");
        executeBenchmarkRun(device, rank, world_size, 100, 0, 1024, false, false);
        for (auto m : part_sz_vec) {
            size_t time = executeBenchmarkRun(device, rank, world_size, 5, 100, m);
            if (rank == 0) {
                RTP_LLM_LOG_INFO("[Custom AR] %d size,%d us", m, time);
            }
            custom_ar_times.push_back(time);
            part_custom_ar_times.push_back(time);
        }

        device                                                        = initTestDevices(rank, world_size, port + k);
        device->initParamsRef().hw_kernel_config.ft_disable_custom_ar = true;
        RTP_LLM_LOG_INFO("[NCCL] Start hot run");
        executeBenchmarkRun(device, rank, world_size, 100, 0, 1024, false, false);
        for (auto m : part_sz_vec) {
            size_t time = executeBenchmarkRun(device, rank, world_size, 5, 100, m, false);
            if (rank == 0) {
                RTP_LLM_LOG_INFO("[NCCL] %d size, %d us", m, time);
            }
            nccl_times.push_back(time);
            part_nccl_times.push_back(time);
        }

        if (rank == 0) {
            double avg_speed_up = 0;
            for (size_t i = 0; i < part_sz_vec.size(); i++) {
                RTP_LLM_LOG_INFO("[AR] %d us, [NCCL] %d us, speed up %f x, Data num %d",
                                 part_custom_ar_times[i],
                                 part_nccl_times[i],
                                 (part_nccl_times[i] * 1.0 / part_custom_ar_times[i]) - 1.0,
                                 part_sz_vec[i]);
                avg_speed_up += (part_nccl_times[i] * 1.0 / part_custom_ar_times[i]) - 1.0;
            }

            RTP_LLM_LOG_INFO("Average speed up %lf x", avg_speed_up / part_sz_vec.size());
        }
        k += 10;
    }

    if (rank == 0) {
        double avg_speed_up = 0;
        for (size_t i = 0; i < sz_vec.size(); i++) {
            avg_speed_up += (nccl_times[i] * 1.0 / custom_ar_times[i]) - 1.0;
        }

        RTP_LLM_LOG_INFO("Average speed up %lf x", avg_speed_up / sz_vec.size());
    }
}

void parse_arguments(int argc, char* argv[], int* run_benchmark, size_t* rank, size_t* world_size, int* port) {
    if (argc != 5) {
        RTP_LLM_LOG_INFO("argc %d\n", argc);
        RTP_LLM_LOG_INFO("Usage: %s <run_benchmark> <rank> <world_size> <port>", argv[0]);
        exit(EXIT_FAILURE);
    }

    *run_benchmark = atoi(argv[1]);
    *rank          = (size_t)strtoul(argv[2], NULL, 10);
    *world_size    = (size_t)strtoul(argv[3], NULL, 10);
    *port          = atoi(argv[4]);
}

int main(int argc, char* argv[]) {
    int    run_benchmark = 0;
    size_t rank          = 0;
    size_t world_size    = 0;
    int    port          = 0;

    parse_arguments(argc, argv, &run_benchmark, &rank, &world_size, &port);

    RTP_LLM_LOG_INFO("run_benchmark: %d", run_benchmark);
    RTP_LLM_LOG_INFO("world_size: %zu", world_size);
    RTP_LLM_LOG_INFO("rank: %zu", rank);
    RTP_LLM_LOG_INFO("port: %d", port);

    fflush(stdout);

    if (run_benchmark == 1) {
        benchmark(rank, world_size, port);
    } else {
        baseTest(rank, world_size, port, 896);
    }

    return 0;
}
