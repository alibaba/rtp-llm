#pragma once

#include <string>

namespace autil {

class URLUtil {
public:
    URLUtil();
    ~URLUtil();

public:
    static std::string decode(const std::string &str);
};

} // namespace autil
