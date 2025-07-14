#include <gtest/gtest.h>
#include <random>
#include "rtp_llm/cpp/kernels/eplb/experts_stats_kernels.h"

using namespace std;
using namespace rtp_llm;

using double_vec_t = tuple<vector<int>, vector<int>>;
using triple_vec_t = tuple<vector<int>, vector<int>, vector<int>>;

class EplbKernelTest: public ::testing::Test {
public:
    void SetUp() override {}

    void TearDown() override {}
};

template<typename T>
T* createDeviceBufferFromVector(const vector<T>& vec) {
    T* device_buf;
    cudaMalloc(&device_buf, vec.size() * sizeof(T));
    cudaMemcpy(device_buf, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice);
    return device_buf;
}

template<typename T>
vector<T> createVectorFromDeviceBuffer(T* device_buf, int size) {
    vector<T> vec(size);
    cudaMemcpy(vec.data(), device_buf, size * sizeof(T), cudaMemcpyDeviceToHost);
    return vec;
}

template<typename T>
vector<T> generateRandomVector(int size, int max_val) {
    vector<T> vec(size);
    for (int i = 0; i < size; i++) {
        vec[i] = rand() % max_val;
    }
    return vec;
}

double_vec_t generateRandomBalancePlan(int log_exp_num, int phy_exp_num) {
    vector<int> result;

    result.reserve(phy_exp_num);
    for (int i = 0; i < log_exp_num; ++i) {
        result.push_back(i);
    }
    random_device rd;
    mt19937       rng(rd());
    shuffle(result.begin(), result.end(), rng);

    if (phy_exp_num > log_exp_num) {
        uniform_int_distribution<int> dist(0, log_exp_num - 1);
        for (int i = 0; i < phy_exp_num - log_exp_num; ++i) {
            result.push_back(dist(rng));
        }
        shuffle(result.begin(), result.end(), rng);
    }

    vector<int> logic_expert_cnt(log_exp_num, 0);
    vector<int> log2phy(log_exp_num * (phy_exp_num - log_exp_num + 1), -1);

    const int max_expert_cnt = phy_exp_num - log_exp_num + 1;
    for (int i = 0; i < phy_exp_num; i++) {
        int cur_log_id                                                      = result[i];
        log2phy[cur_log_id * max_expert_cnt + logic_expert_cnt[cur_log_id]] = i;
        logic_expert_cnt[cur_log_id]++;
    }

    return {log2phy, logic_expert_cnt};
}

void equalExpertBalanceHost(vector<int>&       experts_ids,
                            vector<int>&       log_stats,
                            const vector<int>& log2phy,
                            const vector<int>& logic_expert_cnt,
                            int                log_exp_num,
                            int                phy_exp_num,
                            int                total_tokens,
                            int                ep_rank) {
    int max_exp_num = phy_exp_num - log_exp_num + 1;
    for (int i = 0; i < total_tokens; ++i) {
        int log_exp_id = experts_ids[i];

        int cnt        = logic_expert_cnt[log_exp_id];
        int idx        = log_exp_id * max_exp_num + (i + ep_rank) % cnt;
        int phy_exp_id = log2phy[idx];
        experts_ids[i] = phy_exp_id;

        log_stats[log_exp_id]++;
    }
}

void updateGpuLoadsHost(
    vector<int>& experts_ids, vector<int>& gpu_loads, int total_tokens, int phy_exp_num, int ep_rank, int ep_size) {
    int experts_per_gpu = phy_exp_num / ep_size;
    for (int i = 0; i < total_tokens; ++i) {
        int expert_id = experts_ids[i];
        int gpu_id    = expert_id / experts_per_gpu;
        if (gpu_id == ep_rank) {
            gpu_loads[ep_rank]++;
        }
    }
}

void updateGpuLoadsLLHost(vector<int>& experts_cnts, vector<int>& gpu_loads, int local_experts_num, int ep_rank) {
    for (int i = 0; i < local_experts_num; ++i) {
        gpu_loads[ep_rank] += experts_cnts[i];
    }
}

void updateGpuLoadsDeepEP(vector<int64_t>& expert_ids, vector<int>& gpu_loads, int total_tokens, int ep_rank) {
    for (int i = 0; i < total_tokens; ++i) {
        if (expert_ids[i] >= 0) {
            gpu_loads[ep_rank]++;
        }
    }
}

void equalExpertBalanceTest(int total_tokens, int log_exp_num, int phy_exp_num, int ep_rank, int ep_size) {
    vector<int> experts_ids = generateRandomVector<int>(total_tokens, log_exp_num);
    vector<int> log_stats(log_exp_num, 0);
    auto [log2phy, logic_expert_cnt] = generateRandomBalancePlan(log_exp_num, phy_exp_num);

    int* experts_ids_d      = createDeviceBufferFromVector(experts_ids);
    int* log_stats_d        = createDeviceBufferFromVector(log_stats);
    int* log2phy_d          = createDeviceBufferFromVector(log2phy);
    int* logic_expert_cnt_d = createDeviceBufferFromVector(logic_expert_cnt);

    // device
    cudaStream_t stream = cudaStreamDefault;
    launch_equal_expert_balance(experts_ids_d,
                                log_stats_d,
                                log2phy_d,
                                logic_expert_cnt_d,
                                log_exp_num,
                                phy_exp_num,
                                total_tokens,
                                ep_rank,
                                stream);
    cudaStreamSynchronize(stream);

    // copy back to host
    vector<int> expert_ids_res = createVectorFromDeviceBuffer(experts_ids_d, total_tokens);
    vector<int> log_stats_res  = createVectorFromDeviceBuffer(log_stats_d, log_exp_num);

    // host
    equalExpertBalanceHost(
        experts_ids, log_stats, log2phy, logic_expert_cnt, log_exp_num, phy_exp_num, total_tokens, ep_rank);

    EXPECT_EQ(expert_ids_res, experts_ids);
    EXPECT_EQ(log_stats_res, log_stats);
}

void updateGpuLoadsTest(int total_tokens, int log_exp_num, int phy_exp_num, int ep_rank, int ep_size) {
    vector<int> experts_ids = generateRandomVector<int>(total_tokens, phy_exp_num);
    vector<int> gpu_loads(ep_size, 0);

    int* experts_ids_d = createDeviceBufferFromVector(experts_ids);
    int* gpu_loads_d   = createDeviceBufferFromVector(gpu_loads);

    // device
    cudaStream_t stream = cudaStreamDefault;
    launch_update_gpu_loads(experts_ids_d, gpu_loads_d, total_tokens, phy_exp_num, ep_rank, ep_size, stream);
    cudaStreamSynchronize(stream);

    // copy back to host
    vector<int> gpu_loads_res = createVectorFromDeviceBuffer(gpu_loads_d, ep_size);

    // host
    updateGpuLoadsHost(experts_ids, gpu_loads, total_tokens, phy_exp_num, ep_rank, ep_size);
    EXPECT_EQ(gpu_loads_res, gpu_loads);
}

void updateGpuLoadsLLTest(int local_experts_num, int ep_rank, int ep_size) {
    vector<int> experts_cnts = generateRandomVector<int>(local_experts_num, 10000);
    vector<int> gpu_loads(ep_size, 0);

    int* experts_cnts_d = createDeviceBufferFromVector(experts_cnts);
    int* gpu_loads_d    = createDeviceBufferFromVector(gpu_loads);

    // device
    cudaStream_t stream = cudaStreamDefault;
    launch_update_gpu_loads_ll(experts_cnts_d, gpu_loads_d, local_experts_num, ep_rank, stream);
    cudaStreamSynchronize(stream);

    // copy back to host
    vector<int> gpu_loads_res = createVectorFromDeviceBuffer(gpu_loads_d, ep_size);

    // host
    updateGpuLoadsLLHost(experts_cnts, gpu_loads, local_experts_num, ep_rank);
    EXPECT_EQ(gpu_loads_res, gpu_loads);
}

void updateGpuLoadsDeepEPTest(int total_tokens, int log_exp_num, int phy_exp_num, int ep_rank, int ep_size) {
    int             local_experts_num = phy_exp_num / ep_size;
    vector<int64_t> expert_ids        = generateRandomVector<int64_t>(total_tokens, local_experts_num + 1);
    for (auto& id : expert_ids) {
        id--;
    }
    vector<int> gpu_loads(ep_size, 0);

    int64_t* expert_ids_d = createDeviceBufferFromVector(expert_ids);
    int*     gpu_loads_d  = createDeviceBufferFromVector(gpu_loads);

    // device
    cudaStream_t stream = cudaStreamDefault;
    update_gpu_loads_deepep_kernel(expert_ids_d, gpu_loads_d, total_tokens, ep_rank, stream);
    cudaStreamSynchronize(stream);

    // copy back to host
    vector<int> gpu_loads_res = createVectorFromDeviceBuffer(gpu_loads_d, ep_size);

    // host
    updateGpuLoadsDeepEP(expert_ids, gpu_loads, total_tokens, ep_rank);
    EXPECT_EQ(gpu_loads_res, gpu_loads);
}

TEST_F(EplbKernelTest, equalExpertBalanceTestNoRedundancy) {
    equalExpertBalanceTest(1, 256, 256, 0, 16);
    equalExpertBalanceTest(1, 256, 256, 7, 16);
    equalExpertBalanceTest(8, 256, 256, 0, 16);
    equalExpertBalanceTest(8, 256, 256, 7, 16);
    equalExpertBalanceTest(1024, 256, 256, 0, 16);
    equalExpertBalanceTest(1024, 256, 256, 7, 16);
}

TEST_F(EplbKernelTest, equalExpertBalanceTestRedundancy) {
    equalExpertBalanceTest(1, 256, 288, 0, 16);
    equalExpertBalanceTest(1, 256, 288, 7, 16);
    equalExpertBalanceTest(8, 256, 288, 0, 16);
    equalExpertBalanceTest(8, 256, 288, 7, 16);
    equalExpertBalanceTest(1024, 256, 288, 0, 16);
    equalExpertBalanceTest(1024, 256, 288, 7, 16);
}

TEST_F(EplbKernelTest, updateGpuLoadsTestNoRedundancy) {
    updateGpuLoadsTest(1, 256, 256, 0, 16);
    updateGpuLoadsTest(1, 256, 256, 7, 16);
    updateGpuLoadsTest(8, 256, 256, 0, 16);
    updateGpuLoadsTest(8, 256, 256, 7, 16);
    updateGpuLoadsTest(1024, 256, 256, 0, 16);
    updateGpuLoadsTest(1024, 256, 256, 7, 16);
}

TEST_F(EplbKernelTest, updateGpuLoadsTestRedundancy) {
    updateGpuLoadsTest(1, 256, 288, 0, 16);
    updateGpuLoadsTest(1, 256, 288, 7, 16);
    updateGpuLoadsTest(8, 256, 288, 0, 16);
    updateGpuLoadsTest(8, 256, 288, 7, 16);
    updateGpuLoadsTest(1024, 256, 288, 0, 16);
    updateGpuLoadsTest(1024, 256, 288, 7, 16);
}

TEST_F(EplbKernelTest, updateGpuLoadsLLTest) {
    updateGpuLoadsLLTest(1, 0, 16);
    updateGpuLoadsLLTest(1, 7, 16);
    updateGpuLoadsLLTest(2, 0, 16);
    updateGpuLoadsLLTest(2, 7, 16);
    updateGpuLoadsLLTest(65, 0, 16);
    updateGpuLoadsLLTest(65, 7, 16);
}

TEST_F(EplbKernelTest, updateGpuLoadsDeepEPTestNoRedundancy) {
    updateGpuLoadsDeepEPTest(1, 256, 256, 0, 16);
    updateGpuLoadsDeepEPTest(1, 256, 256, 7, 16);
    updateGpuLoadsDeepEPTest(8, 256, 256, 0, 16);
    updateGpuLoadsDeepEPTest(8, 256, 256, 7, 16);
    updateGpuLoadsDeepEPTest(1024, 256, 256, 0, 16);
    updateGpuLoadsDeepEPTest(1024, 256, 256, 7, 16);
}

TEST_F(EplbKernelTest, updateGpuLoadsDeepEPTestRedundancy) {
    updateGpuLoadsDeepEPTest(1, 256, 288, 0, 16);
    updateGpuLoadsDeepEPTest(1, 256, 288, 7, 16);
    updateGpuLoadsDeepEPTest(8, 256, 288, 0, 16);
    updateGpuLoadsDeepEPTest(8, 256, 288, 7, 16);
    updateGpuLoadsDeepEPTest(1024, 256, 288, 0, 16);
    updateGpuLoadsDeepEPTest(1024, 256, 288, 7, 16);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}