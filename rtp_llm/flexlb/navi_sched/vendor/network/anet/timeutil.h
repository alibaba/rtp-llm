#pragma once
#include <stdint.h>
namespace anet {

class TimeUtil {
public:
    /*
     * 得到当前时间
     */
    static int64_t getTime();

    /*
     * 设置当前时间到now中
     */
    static void setNow();

    /*
     * convert a time usec value to a date str.
     * buf should have at least 26 slot in size.
     */
    static void getTimeStr(int64_t tval, char *buf);

public:
    static int64_t _now;
    static const int64_t MIN;
    static const int64_t MAX;
    static const int64_t PRE_MAX;
};

} // namespace anet

