/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/cuda/cuda_utils.h"
#include "cuda_fp8_utils.h"

#include <stdio.h>
#include <stdlib.h>

namespace fastertransformer {

/* **************************** debug tools ********************************* */

template<typename T>
void print_to_file(
    const T* result, const int size, const char* file, cudaStream_t stream, std::ios::openmode open_mode) {
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    printf("[INFO] file: %s with size %d.\n", file, size);
    std::ofstream outFile(file, open_mode);
    if (outFile) {
        T* tmp = new T[size];
        check_cuda_error(cudaMemcpyAsync(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost, stream));
        for (int i = 0; i < size; ++i) {
            float val = (float)(tmp[i]);
            outFile << val << std::endl;
        }
        delete[] tmp;
    } else {
        throw std::runtime_error(std::string("[FT][ERROR] Cannot open file: ") + file + "\n");
    }
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
}

template void
print_to_file(const float* result, const int size, const char* file, cudaStream_t stream, std::ios::openmode open_mode);
template void
print_to_file(const half* result, const int size, const char* file, cudaStream_t stream, std::ios::openmode open_mode);
#ifdef ENABLE_BF16
template void print_to_file(
    const __nv_bfloat16* result, const int size, const char* file, cudaStream_t stream, std::ios::openmode open_mode);
#endif

template<typename T>
void print_abs_mean(const T* buf, uint size, cudaStream_t stream, std::string name) {
    if (buf == nullptr) {
        FT_LOG_WARNING("It is an nullptr, skip!");
        return;
    }
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    T* h_tmp = new T[size];
    cudaMemcpyAsync(h_tmp, buf, sizeof(T) * size, cudaMemcpyDeviceToHost, stream);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    double   sum        = 0.0f;
    uint64_t zero_count = 0;
    float    max_val    = -1e10;
    bool     find_inf   = false;
    for (uint i = 0; i < size; i++) {
        if (std::isinf((float)(h_tmp[i]))) {
            find_inf = true;
            continue;
        }
        sum += abs((double)h_tmp[i]);
        if ((float)h_tmp[i] == 0.0f) {
            zero_count++;
        }
        max_val = max_val > abs(float(h_tmp[i])) ? max_val : abs(float(h_tmp[i]));
    }
    printf("[INFO][FT] %20s size: %u, abs mean: %f, abs sum: %f, abs max: %f, find inf: %s",
           name.c_str(),
           size,
           sum / size,
           sum,
           max_val,
           find_inf ? "true" : "false");
    std::cout << std::endl;
    delete[] h_tmp;
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
}

template void print_abs_mean(const float* buf, uint size, cudaStream_t stream, std::string name);
template void print_abs_mean(const half* buf, uint size, cudaStream_t stream, std::string name);
#ifdef ENABLE_BF16
template void print_abs_mean(const __nv_bfloat16* buf, uint size, cudaStream_t stream, std::string name);
#endif
template void print_abs_mean(const int* buf, uint size, cudaStream_t stream, std::string name);
template void print_abs_mean(const uint* buf, uint size, cudaStream_t stream, std::string name);
template void print_abs_mean(const int8_t* buf, uint size, cudaStream_t stream, std::string name);
#ifdef ENABLE_FP8
template void print_abs_mean(const __nv_fp8_e4m3* buf, uint size, cudaStream_t stream, std::string name);
#endif

template<typename T>
void print_to_screen(const T* result, const int size) {
    if (result == nullptr) {
        FT_LOG_WARNING("It is an nullptr, skip! \n");
        return;
    }
    T* tmp = reinterpret_cast<T*>(malloc(sizeof(T) * size));
    check_cuda_error(cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        printf("%d, %f\n", i, static_cast<float>(tmp[i]));
    }
    free(tmp);
}

template void print_to_screen(const float* result, const int size);
template void print_to_screen(const half* result, const int size);
#ifdef ENABLE_BF16
template void print_to_screen(const __nv_bfloat16* result, const int size);
#endif
template void print_to_screen(const int* result, const int size);
template void print_to_screen(const uint* result, const int size);
template void print_to_screen(const bool* result, const int size);
#ifdef ENABLE_FP8
template void print_to_screen(const __nv_fp8_e4m3* result, const int size);
#endif

template<typename T>
void printMatrix(T* ptr, int m, int k, int stride, bool is_device_ptr) {
    T* tmp;
    if (is_device_ptr) {
        // k < stride ; stride = col-dimension.
        tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
        check_cuda_error(cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    } else {
        tmp = ptr;
    }

    for (int ii = -1; ii < m; ++ii) {
        if (ii >= 0) {
            printf("%02d ", ii);
        } else {
            printf("   ");
        }

        for (int jj = 0; jj < k; jj += 1) {
            if (ii >= 0) {
                printf("%7.3f ", (float)tmp[ii * stride + jj]);
            } else {
                printf("%7d ", jj);
            }
        }
        printf("\n");
    }
    if (is_device_ptr) {
        free(tmp);
    }
}

template void printMatrix(float* ptr, int m, int k, int stride, bool is_device_ptr);
template void printMatrix(half* ptr, int m, int k, int stride, bool is_device_ptr);
#ifdef ENABLE_BF16
template void printMatrix(__nv_bfloat16* ptr, int m, int k, int stride, bool is_device_ptr);
#endif

void printMatrix(unsigned long long* ptr, int m, int k, int stride, bool is_device_ptr) {
    typedef unsigned long long T;
    T*                         tmp;
    if (is_device_ptr) {
        // k < stride ; stride = col-dimension.
        tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
        check_cuda_error(cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    } else {
        tmp = ptr;
    }

    for (int ii = -1; ii < m; ++ii) {
        if (ii >= 0) {
            printf("%02d ", ii);
        } else {
            printf("   ");
        }

        for (int jj = 0; jj < k; jj += 1) {
            if (ii >= 0) {
                printf("%4llu ", tmp[ii * stride + jj]);
            } else {
                printf("%4d ", jj);
            }
        }
        printf("\n");
    }
    if (is_device_ptr) {
        free(tmp);
    }
}

void printMatrix(int* ptr, int m, int k, int stride, bool is_device_ptr) {
    typedef int T;
    T*          tmp;
    if (is_device_ptr) {
        // k < stride ; stride = col-dimension.
        tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
        check_cuda_error(cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    } else {
        tmp = ptr;
    }

    for (int ii = -1; ii < m; ++ii) {
        if (ii >= 0) {
            printf("%02d ", ii);
        } else {
            printf("   ");
        }

        for (int jj = 0; jj < k; jj += 1) {
            if (ii >= 0) {
                printf("%4d ", tmp[ii * stride + jj]);
            } else {
                printf("%4d ", jj);
            }
        }
        printf("\n");
    }
    if (is_device_ptr) {
        free(tmp);
    }
}

void printMatrix(size_t* ptr, int m, int k, int stride, bool is_device_ptr) {
    typedef size_t T;
    T*             tmp;
    if (is_device_ptr) {
        // k < stride ; stride = col-dimension.
        tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
        check_cuda_error(cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    } else {
        tmp = ptr;
    }

    for (int ii = -1; ii < m; ++ii) {
        if (ii >= 0) {
            printf("%02d ", ii);
        } else {
            printf("   ");
        }

        for (int jj = 0; jj < k; jj += 1) {
            if (ii >= 0) {
                printf("%4ld ", tmp[ii * stride + jj]);
            } else {
                printf("%4d ", jj);
            }
        }
        printf("\n");
    }
    if (is_device_ptr) {
        free(tmp);
    }
}

template<typename T>
void check_max_val(const T* result, const int size) {
    T* tmp = new T[size];
    cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost);
    float max_val = -100000;
    for (int i = 0; i < size; i++) {
        float val = static_cast<float>(tmp[i]);
        if (val > max_val) {
            max_val = val;
        }
    }
    delete tmp;
    printf("[INFO][CUDA] addr %p max val: %f \n", result, max_val);
}

template void check_max_val(const float* result, const int size);
template void check_max_val(const half* result, const int size);
#ifdef ENABLE_BF16
template void check_max_val(const __nv_bfloat16* result, const int size);
#endif

template<typename T>
void check_abs_mean_val(const T* result, const int size) {
    T* tmp = new T[size];
    cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += abs(static_cast<float>(tmp[i]));
    }
    delete tmp;
    printf("[INFO][CUDA] addr %p abs mean val: %f \n", result, sum / size);
}

template void check_abs_mean_val(const float* result, const int size);
template void check_abs_mean_val(const half* result, const int size);
#ifdef ENABLE_BF16
template void check_abs_mean_val(const __nv_bfloat16* result, const int size);
#endif

/* ***************************** common utils ****************************** */

cudaError_t getSetDevice(int i_device, int* o_device) {
    int         current_dev_id = 0;
    cudaError_t err            = cudaSuccess;

    if (o_device != NULL) {
        err = cudaGetDevice(&current_dev_id);
        if (err != cudaSuccess) {
            return err;
        }
        if (current_dev_id == i_device) {
            *o_device = i_device;
        } else {
            err = cudaSetDevice(i_device);
            if (err != cudaSuccess) {
                return err;
            }
            *o_device = current_dev_id;
        }
    } else {
        err = cudaSetDevice(i_device);
        if (err != cudaSuccess) {
            return err;
        }
    }

    return cudaSuccess;
}



/*
  b = batch_szie
  s = seq_len
  h = head_num
  d = hidden_per_head/hidden
 */
template<typename T>
void print_bshd(const int   layer_id,
                const char* name,
                const T*    ptr,
                int         batch_size,
                int         seq_len,
                int         num_heads,
                int         hidden_size_per_head,
                bool        is_device_ptr) {
    if (!should_print()) {
        return;
    }

    T* cpu_ptr = nullptr;
    if (is_device_ptr) {
        cudaDeviceSynchronize();
        auto size = batch_size * seq_len * num_heads * hidden_size_per_head * sizeof(T);
        cpu_ptr   = reinterpret_cast<T*>(malloc(size));
        check_cuda_error(cudaMemcpy(cpu_ptr, ptr, size, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    } else {
        cpu_ptr = const_cast<T*>(ptr);
    }
    printf("layer_id: %d %s [%d %d %d %d]\n", layer_id, name, batch_size, seq_len, num_heads, hidden_size_per_head);
    auto md_array_ptr = (T(*)[seq_len][num_heads][hidden_size_per_head])cpu_ptr;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < std::min(num_heads, 4); k++) {
                printf("b_%d s_%d h_%d %f %f %f %f\n",
                       i,
                       j,
                       k,
                       float(md_array_ptr[i][j][k][0]),
                       float(md_array_ptr[i][j][k][1]),
                       float(md_array_ptr[i][j][k][2]),
                       float(md_array_ptr[i][j][k][3]));
            }
        }
    }
}

template<typename T>
void print_bhsd(const int   layer_id,
                const char* name,
                const T*    ptr,
                int         batch_size,
                int         num_heads,
                int         seq_len,
                int         hidden_size_per_head,
                bool        is_device_ptr) {
    if (!should_print()) {
        return;
    }
    T* cpu_ptr = nullptr;
    if (is_device_ptr) {
        cudaDeviceSynchronize();
        auto size = batch_size * seq_len * num_heads * hidden_size_per_head * sizeof(T);
        cpu_ptr   = reinterpret_cast<T*>(malloc(size));
        check_cuda_error(cudaMemcpy(cpu_ptr, ptr, size, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    } else {
        cpu_ptr = const_cast<T*>(ptr);
    }
    printf("layer_id: %d %s [%d %d %d %d]\n", layer_id, name, batch_size, num_heads, seq_len, hidden_size_per_head);
    auto md_array_ptr = (T(*)[num_heads][seq_len][hidden_size_per_head])cpu_ptr;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < std::min(num_heads, 4); k++) {
                printf("b_%d s_%d h_%d %f %f %f %f\n",
                       i,
                       j,
                       k,
                       float(md_array_ptr[i][k][j][0]),
                       float(md_array_ptr[i][k][j][1]),
                       float(md_array_ptr[i][k][j][2]),
                       float(md_array_ptr[i][k][j][3]));
            }
        }
    }
}

template<typename T>
void print_bhss(const int   layer_id,
                const char* name,
                const T*    ptr,
                int         batch_size,
                int         num_heads,
                int         seq_len,
                int         seq_len2,
                bool        is_device_ptr) {
    if (!should_print()) {
        return;
    }
    T* cpu_ptr = nullptr;
    if (is_device_ptr) {
        cudaDeviceSynchronize();
        uint64_t size = (uint64_t)batch_size * (uint64_t)num_heads * (uint64_t)seq_len * (uint64_t)seq_len2 * sizeof(T);
        cpu_ptr       = reinterpret_cast<T*>(malloc(size));
        check_cuda_error(cudaMemcpy(cpu_ptr, ptr, size, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    } else {
        cpu_ptr = const_cast<T*>(ptr);
    }
    printf("layer_id: %d %s [%d %d %d %d]\n", layer_id, name, batch_size, num_heads, seq_len, seq_len2);

    auto md_array_ptr = (T(*)[num_heads][seq_len][seq_len2])cpu_ptr;
    for (int i = 0; i < batch_size; i++) {
        for (int head = 0; head < std::min(num_heads, 4); head++) {
            for (int j1 = 0; j1 < seq_len; j1++) {
                for (int j2 = 0; j2 < seq_len2; j2++) {
                    printf("b_%d h_%d s_%d_%d %f \n", i, head, j1, j2, float(md_array_ptr[i][head][j1][j2]));
                }
            }
        }
    }
}

template<typename T>
void print_bsd(const int   layer_id,
               const char* name,
               const T*    ptr,
               int         batch_size,
               int         seq_len,
               int         hidden_size,
               int         start,
               int         end,
               bool        is_device_ptr) {
    if (!should_print()) {
        return;
    }
    T* cpu_ptr = nullptr;
    if (is_device_ptr) {
        cudaDeviceSynchronize();
        auto size = batch_size * hidden_size * seq_len * sizeof(T);
        cpu_ptr   = reinterpret_cast<T*>(malloc(size));
        check_cuda_error(cudaMemcpy(cpu_ptr, ptr, size, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    } else {
        cpu_ptr = const_cast<T*>(ptr);
    }
    printf("layer_id: %d %s [%d %d %d]\n", layer_id, name, batch_size, seq_len, hidden_size);
    auto md_array_ptr = (T(*)[seq_len][hidden_size])cpu_ptr;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            printf("b_%d s_%d ", i, j);
            double sum1 = 0;
            double sum2 = 0;
            for (int k = start; k < end; k++) {
                printf("k = %d, value = %f ", k, float(md_array_ptr[i][j][k]));
                sum1 += float(md_array_ptr[i][j][k]);
                sum2 += float(md_array_ptr[i][j][k]) * float(md_array_ptr[i][j][k]);
            }
            printf("\n");
            printf("sum1 = %f, square sum2 = %lf\n", sum1, sum2);
        }
    }
}

template<typename T>
void print_kv_cache(const int   layer_id,
                    const char* name,
                    const T*    ptr,
                    int         dim1,
                    int         dim2,
                    int         dim3,
                    int         dim4,
                    int         dim5,
                    int         dim6,
                    bool        print_all,
                    bool        is_device_ptr) {
    if (!should_print()) {
        return;
    }

    std::cout.setf(std::ios::fixed);

    T* cpu_ptr = nullptr;
    if (is_device_ptr) {
        cpu_ptr = reinterpret_cast<T*>(malloc(dim1 * dim2 * dim3 * dim4 * dim5 * dim6 * sizeof(T)));
        check_cuda_error(
            cudaMemcpy(cpu_ptr, ptr, dim1 * dim2 * dim3 * dim4 * dim5 * dim6 * sizeof(T), cudaMemcpyDeviceToHost));
    } else {
        cpu_ptr = const_cast<T*>(ptr);
    }

    printf("layer_id: %d %s [%d %d %d %d %d %d]\n", layer_id, name, dim1, dim2, dim3, dim4, dim5, dim6);
    fflush(stdout);
    for (uint64_t i = 0; i < dim1; i++) {
        for (uint64_t j = 0; j < dim2; j++) {
            for (uint64_t k = 0; k < dim3; k++) {
                for (uint64_t x = 0; x < dim4; x++) {
                    for (uint64_t y = 0; y < dim5; y++) {
                        printf("i = %d, j = %d, k = %d, x = %d, y = %d\n", i, j, k, x, y);
                        if (print_all) {
                            for (uint64_t z = 0; z < dim6; z++) {
                                std::cout << float(*(cpu_ptr + i * dim2 * dim3 * dim4 * dim5 * dim6
                                                     + j * dim3 * dim4 * dim5 * dim6 + k * dim4 * dim5 * dim6
                                                     + x * dim5 * dim6 + y * dim6 + z))
                                          << ", ";
                            }
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
                printf("\n\n");
            }
            printf("\n\n");
        }
        printf("\n\n");
    }
    printf("\n\n");
}

template<typename T>
void print_bsd_sum_and_square(const int   layer_id,
                              const char* name,
                              const T*    ptr,
                              int         batch_size,
                              int         seq_len,
                              int         hidden_size,
                              int         start,
                              int         end,
                              bool        is_device_ptr) {
    if (!should_print()) {
        return;
    }

    static char* tp_rank = std::getenv("WORLD_RANK");
    if (tp_rank && (strcmp(tp_rank, "0") != 0 && strcmp(tp_rank, "1") != 0)) {
        return;
    }

    T* cpu_ptr = nullptr;
    if (is_device_ptr) {
        cudaDeviceSynchronize();
        auto size = batch_size * hidden_size * seq_len * sizeof(T);
        cpu_ptr   = reinterpret_cast<T*>(malloc(size));
        check_cuda_error(cudaMemcpy(cpu_ptr, ptr, size, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    } else {
        cpu_ptr = const_cast<T*>(ptr);
    }
    auto md_array_ptr = (T(*)[seq_len][hidden_size])cpu_ptr;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            double sum1 = 0;
            double sum2 = 0;
            for (int k = start; k < end; k++) {
                printf("layer_id: %d %s [%d %d %d %d], rank = %s, k = %d, value = %f ",
                       layer_id,
                       name,
                       batch_size,
                       seq_len,
                       hidden_size,
                       j,
                       tp_rank,
                       k,
                       float(md_array_ptr[i][j][k]));
                sum1 += float(md_array_ptr[i][j][k]);
                sum2 += float(md_array_ptr[i][j][k]) * float(md_array_ptr[i][j][k]);
            }
            printf("\nlayer_id: %d %s [%d %d %d %d], rank = %s, sum1 = %f, square sum2 = %lf\n",
                   layer_id,
                   name,
                   batch_size,
                   seq_len,
                   hidden_size,
                   j,
                   tp_rank,
                   sum1,
                   sum2);
        }
    }
}

#define DECLARE_PRINT_TYPE(CPP_TYPE)                                                                                   \
    template void print_bhsd<CPP_TYPE>(const int       layer_id,                                                       \
                                       const char*     name,                                                           \
                                       const CPP_TYPE* ptr,                                                            \
                                       int             batch_size,                                                     \
                                       int             num_heads,                                                      \
                                       int             seq_len,                                                        \
                                       int             hidden_size_per_head,                                           \
                                       bool            is_device_ptr);                                                            \
    template void print_bshd<CPP_TYPE>(const int       layer_id,                                                       \
                                       const char*     name,                                                           \
                                       const CPP_TYPE* ptr,                                                            \
                                       int             batch_size,                                                     \
                                       int             seq_len,                                                        \
                                       int             num_heads,                                                      \
                                       int             hidden_size_per_head,                                           \
                                       bool            is_device_ptr);                                                            \
    template void print_bhss<CPP_TYPE>(const int       layer_id,                                                       \
                                       const char*     name,                                                           \
                                       const CPP_TYPE* ptr,                                                            \
                                       int             batch_size,                                                     \
                                       int             num_heads,                                                      \
                                       int             seq_len,                                                        \
                                       int             seq_len2,                                                       \
                                       bool            is_device_ptr);                                                            \
    template void print_bsd<CPP_TYPE>(const int       layer_id,                                                        \
                                      const char*     name,                                                            \
                                      const CPP_TYPE* ptr,                                                             \
                                      int             batch_size,                                                      \
                                      int             seq_len,                                                         \
                                      int             hidden_size,                                                     \
                                      int             start,                                                           \
                                      int             end,                                                             \
                                      bool            is_device_ptr);                                                             \
    template void print_bsd_sum_and_square<CPP_TYPE>(const int       layer_id,                                         \
                                                     const char*     name,                                             \
                                                     const CPP_TYPE* ptr,                                              \
                                                     int             batch_size,                                       \
                                                     int             seq_len,                                          \
                                                     int             hidden_size,                                      \
                                                     int             start,                                            \
                                                     int             end,                                              \
                                                     bool            is_device_ptr);                                              \
    template void print_kv_cache<CPP_TYPE>(const int       layer_id,                                                   \
                                           const char*     name,                                                       \
                                           const CPP_TYPE* ptr,                                                        \
                                           int             dim1,                                                       \
                                           int             dim2,                                                       \
                                           int             dim3,                                                       \
                                           int             dim4,                                                       \
                                           int             dim5,                                                       \
                                           int             dim6,                                                       \
                                           bool            print_all,                                                  \
                                           bool            is_device_ptr);

DECLARE_PRINT_TYPE(float);
DECLARE_PRINT_TYPE(half);
DECLARE_PRINT_TYPE(__nv_bfloat16);
DECLARE_PRINT_TYPE(int8_t);
DECLARE_PRINT_TYPE(uint8_t);
DECLARE_PRINT_TYPE(int);
DECLARE_PRINT_TYPE(int64_t);

/* ************************** end of common utils ************************** */
}  // namespace fastertransformer
