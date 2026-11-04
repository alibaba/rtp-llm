#include "autil/metric/MetricUtil.h"

#include <ctype.h>
#include <string.h>

namespace autil {
namespace metric {

MetricUtil::MetricUtil() {}

MetricUtil::~MetricUtil() {}

const char *MetricUtil::skipToken(const char *str) {
    unsigned char *p = (unsigned char *)str;
    while (isspace(*p)) {
        p++;
    }
    while (*p && !isspace(*p)) {
        p++;
    }
    return (char *)p;
}

const char *MetricUtil::skipWhite(const char *str) {
    unsigned char *p = (unsigned char *)str;
    int i = strlen(str);
    while (i > 0 && isspace(*p)) {
        p++;
        i--;
    }
    if (i == 0) {
        return NULL;
    }
    return (char *)p;
}

} // namespace metric
} // namespace autil
