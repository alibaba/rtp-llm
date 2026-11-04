#pragma once

#include <memory>

namespace autil {
namespace metric {

class MetricUtil {
public:
    MetricUtil();
    ~MetricUtil();

private:
    MetricUtil(const MetricUtil &);
    MetricUtil &operator=(const MetricUtil &);

public:
    static const char *skipToken(const char *str);
    static const char *skipWhite(const char *str);
};

typedef std::shared_ptr<MetricUtil> MetricUtilPtr;

} // namespace metric
} // namespace autil
