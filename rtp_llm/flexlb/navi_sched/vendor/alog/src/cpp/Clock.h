#pragma once
/**
 * @file     Clock.h
 * @brief    Utility class to get the current time of different precision.
 *
 * @version  1.0.0
 * @date     2013.1.12
 * @author   yuejun.huyj
 */

#include <stdint.h>
#include <sys/time.h>

namespace alog {

class Clock {
public:
    Clock(){};
    ~Clock(){};

    // Get the current time in miliseconds
    static uint64_t nowMS() {
        struct timeval tv = {0};
        gettimeofday(&tv, NULL);
        return (tv.tv_sec * (uint64_t)1000 + tv.tv_usec / 1000);
    }

    // Get the current time in microseconds
    static uint64_t nowUS() {
        struct timeval tv = {0};
        gettimeofday(&tv, NULL);
        return (tv.tv_sec * (uint64_t)1000000 + tv.tv_usec);
    }

private:
    Clock(const Clock &);
    Clock &operator=(const Clock &);
};

} // namespace alog
