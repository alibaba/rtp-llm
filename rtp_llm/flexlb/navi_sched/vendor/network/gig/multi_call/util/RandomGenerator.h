#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#if !__aarch64__
#include <immintrin.h>
#endif

namespace multi_call {

class RandomGenerator
{
public:
    RandomGenerator(uint64_t randomSeed) : _randomSeed(randomSeed) {
        _supportHWRandom = supportRdRand();
    }
    ~RandomGenerator() {
    }

private:
    RandomGenerator(const RandomGenerator &);
    RandomGenerator &operator=(const RandomGenerator &);

public:
    void init(uint64_t randomSeed) {
        _randomSeed = randomSeed;
    }
    uint64_t get();

public:
    static void setUseHDRandom(bool value) {
        _useHWRandom = value;
    }
    static bool useHDRandom() {
        return _supportHWRandom && _useHWRandom;
    }

private:
    static bool supportRdRand();

private:
    volatile uint64_t _randomSeed;

private:
    static bool _supportHWRandom;
    static bool _useHWRandom;

private:
    static const int32_t RETRY_COUNT;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(RandomGenerator);

} // namespace multi_call

