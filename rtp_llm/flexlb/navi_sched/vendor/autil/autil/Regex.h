#pragma once

#include <string>
#include <vector>

namespace autil {

class Regex {
private:
    Regex();
    ~Regex();

public:
    static bool match(const std::string &str, const std::string &pattern);
    static bool match(const std::string &str, const std::string &pattern, int regex_mode);
    static bool match(const char *str, const std::string &pattern);
    static bool match(const char *str, const std::string &pattern, int regex_mode);
    static bool groupMatch(const std::string &str, const std::string &pattern, std::vector<std::string> &matched_items);
};

} // namespace autil
