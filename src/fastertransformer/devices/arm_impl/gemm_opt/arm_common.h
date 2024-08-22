/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    arm_common.h
 */

#pragma once
#include <iostream>
#include <vector>
#define RTP_RUNTIME_THREAD 1
#define RTP_OMP 1
#if RTP_RUNTIME_THREAD == RTP_OMP
#include <omp.h>
#endif

#define RTP_OPENMP_MANUAL_STATIC_SPLIT 0  // if use manual split, other wise, use openmp native method.

namespace fastertransformer {

#if RTP_RUNTIME_THREAD == RTP_OMP
inline int get_max_threads() {
    return omp_get_max_threads();
}
#endif

// https://github.com/opencv/dldt/blob/2019/inference-engine/include/ie_parallel.hpp#L285
template<typename T>
static inline T parallel_it_init(T start) {
    return start;
}
template<typename T, typename Q, typename R, typename... Args>
static inline T parallel_it_init(T start, Q& x, const R& X, Args&&... tuple) {
    start = parallel_it_init(start, static_cast<Args>(tuple)...);
    x     = start % X;
    return start / X;
}

inline bool parallel_it_step() {
    return true;
}

template<typename Q, typename R, typename... Args>
inline bool parallel_it_step(Q& x, const R& X, Args&&... tuple) {
    if (parallel_it_step(static_cast<Args>(tuple)...)) {
        x = (x + 1) % X;
        return x == 0;
    }
    return false;
}

static inline int parallel_init(int start, int size, std::vector<int>& counters, std::vector<int>& dims) {
    for (int j = size - 1; j >= 0; j--) {
        counters[j] = start % dims[j];
        start       = start / dims[j];
    }
    return start;
}

static inline void parallel_step(int size, std::vector<int>& counters, std::vector<int>& dims) {
    for (int j = size - 1; j >= 0; j--) {
        counters[j] = (counters[j] + 1) % dims[j];
        if (counters[j] != 0)
            return;
    }
}

template<typename T, typename Q>
inline void splitter(const T& n, const Q& team, const Q& tid, T& n_start, T& n_end) {
    // always let openmp to do the split.
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_end   = n;
    } else {
        T n1    = (n + (T)team - 1) / (T)team;
        T n2    = n1 - 1;
        T T1    = n - n2 * (T)team;
        n_end   = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

// Function f has two inputs, thread_num and num_threads
template<typename F>
void parallel(int work_amount, F f, int nthr) {
    if (nthr > work_amount)
        nthr = work_amount;
    if (nthr == 1) {
        f(0, 1);
        return;
    }
#if RTP_RUNTIME_THREAD == RTP_OMP
#pragma omp parallel num_threads(nthr)
    {
        int ithr = omp_get_thread_num();
        f(ithr, nthr);
    }
#endif
    return;
}

template<typename F>
void parallel(int work_amount, F f) {
#if RTP_RUNTIME_THREAD == RTP_OMP
    if ((work_amount == 1) || omp_in_parallel()) {
        parallel(work_amount, f, 1);
        return;
    }
#endif
    int nthr = get_max_threads();
    parallel(work_amount, f, nthr);
    return;
}

template<typename T0, typename F>
void parallel_for(const T0& D0, const F& func) {
#if RTP_OPENMP_MANUAL_STATIC_SPLIT
    parallel(D0, [&](int ithr, int nthr) {
        T0 start, end;
#if RTP_RUNTIME_THREAD == RTP_OMP
        splitter(D0, nthr, ithr, start, end);
#pragma omp parallel for
#endif
        for (T0 d0 = start; d0 < end; ++d0)
            func(d0);
    });

#else
#if RTP_RUNTIME_THREAD == RTP_OMP
    int nthr = get_max_threads();
#pragma omp parallel for num_threads(nthr)
    for (T0 d0 = 0; d0 < D0; ++d0)
        func(d0);
#endif
#endif
}

template<typename T0, typename T1, typename F>
void parallel_for(const T0& D0, const T1& D1, const F& func) {
#if RTP_OPENMP_MANUAL_STATIC_SPLIT
    const int work_amount = (int)D0 * D1;
    parallel(work_amount, [&](int ithr, int nthr) {
        int start, end;
        splitter(work_amount, nthr, ithr, start, end);
        T0 d0;
        T1 d1;
        parallel_it_init(start, d0, D0, d1, D1);
#if RTP_RUNTIME_THREAD == RTP_OMP
#pragma omp parallel for
#endif
        for (int iwork = start; iwork < end; ++iwork) {
            func(d0, d1);
            parallel_it_step(d0, D0, d1, D1);
        }
    });
#else
#if RTP_RUNTIME_THREAD == RTP_OMP
    int nthr = get_max_threads();
#pragma omp parallel for collapse(2) num_threads(nthr)
    for (T0 d0 = 0; d0 < D0; ++d0) {
        for (T1 d1 = 0; d1 < D1; ++d1) {
            func(d0, d1);
        }
    }
#endif
#endif
}

template<typename T0, typename F>
void parallel_for_num_threads(int num_threads, const T0& D0, const F& func) {
    parallel(
        D0,
        [&](int ithr, int nthr) {
            T0 start, end;
            splitter(D0, nthr, ithr, start, end);
            for (T0 d0 = start; d0 < end; ++d0)
                func(d0);
        },
        num_threads);
}

template<typename T0, typename T1, typename F>
void parallel_for_num_threads(int num_threads, const T0& D0, const T1& D1, const F& func) {
    const int work_amount = (int)D0 * D1;
    parallel(
        work_amount,
        [&](int ithr, int nthr) {
            int start, end;
            splitter(work_amount, nthr, ithr, start, end);
            T0 d0;
            T1 d1;
            parallel_it_init(start, d0, D0, d1, D1);
            for (int iwork = start; iwork < end; ++iwork) {
                func(d0, d1);
                parallel_it_step(d0, D0, d1, D1);
            }
        },
        num_threads);
}

template<typename T0, typename T1, typename T2, typename F>
void parallel_for_num_threads(int num_threads, const T0& D0, const T1& D1, const T2& D2, const F& func) {
    const int work_amount = (int)D0 * D1 * D2;
    parallel(
        work_amount,
        [&](int ithr, int nthr) {
            int start, end;
            splitter(work_amount, nthr, ithr, start, end);
            T0 d0;
            T1 d1;
            T2 d2;
            parallel_it_init(start, d0, D0, d1, D1, d2, D2);
            for (int iwork = start; iwork < end; ++iwork) {
                func(d0, d1, d2);
                parallel_it_step(d0, D0, d1, D1, d2, D2);
            }
        },
        num_threads);
}

template<typename T0, typename T1, typename T2, typename T3, typename F>
void parallel_for_num_threads(int num_threads, const T0& D0, const T1& D1, const T2& D2, const T3& D3, const F& func) {
    const int work_amount = (int)D0 * D1 * D2 * D3;
    parallel(
        work_amount,
        [&](int ithr, int nthr) {
            int start, end;
            splitter(work_amount, nthr, ithr, start, end);
            T0 d0;
            T1 d1;
            T2 d2;
            T3 d3;
            parallel_it_init(start, d0, D0, d1, D1, d2, D2, d3, D3);
            for (int iwork = start; iwork < end; ++iwork) {
                func(d0, d1, d2, d3);
                parallel_it_step(d0, D0, d1, D1, d2, D2, d3, D3);
            }
        },
        num_threads);
}

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename F>
void parallel_for_num_threads(
    int num_threads, const T0& D0, const T1& D1, const T2& D2, const T3& D3, const T4& D4, const F& func) {
    const int work_amount = (int)D0 * D1 * D2 * D3 * D4;
    parallel(
        work_amount,
        [&](int ithr, int nthr) {
            int start, end;
            splitter(work_amount, nthr, ithr, start, end);
            T0 d0;
            T1 d1;
            T2 d2;
            T3 d3;
            T4 d4;
            parallel_it_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
            for (int iwork = start; iwork < end; ++iwork) {
                func(d0, d1, d2, d3, d4);
                parallel_it_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
            }
        },
        num_threads);
}

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename F>
void parallel_for_num_threads(int       num_threads,
                              const T0& D0,
                              const T1& D1,
                              const T2& D2,
                              const T3& D3,
                              const T4& D4,
                              const T5& D5,
                              const F&  func) {
    const int work_amount = (int)D0 * D1 * D2 * D3 * D4 * D5;
    parallel(
        work_amount,
        [&](int ithr, int nthr) {
            int start, end;
            splitter(work_amount, nthr, ithr, start, end);
            T0 d0;
            T1 d1;
            T2 d2;
            T3 d3;
            T4 d4;
            T5 d5;
            parallel_it_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
            for (int iwork = start; iwork < end; ++iwork) {
                func(d0, d1, d2, d3, d4, d5);
                parallel_it_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
            }
        },
        num_threads);
}

}  // namespace fastertransformer
