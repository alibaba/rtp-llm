#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace rtp_llm::benchmark {

using NextArgumentValue = std::function<std::string()>;

template<typename Handler>
void consumeOptions(int& argc, char**& argv, Handler&& handler) {
    int write_index = 1;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument.rfind("--", 0) != 0) {
            argv[write_index++] = argv[index];
            continue;
        }

        const size_t      equal        = argument.find('=');
        const std::string key          = argument.substr(2, equal == std::string::npos ? equal : equal - 2);
        const std::string inline_value = equal == std::string::npos ? "" : argument.substr(equal + 1);
        auto              next         = [&]() {
            if (!inline_value.empty()) {
                return inline_value;
            }
            if (++index < argc) {
                return std::string(argv[index]);
            }
            throw std::runtime_error("Missing value for --" + key);
        };
        if (!handler(key, next)) {
            argv[write_index++] = argv[index];
        }
    }
    argc = write_index;
}

inline uint64_t parseUnsigned(const std::string& key, const NextArgumentValue& next) {
    const std::string text = next();
    size_t            parsed{0};
    uint64_t          value{0};
    try {
        value = std::stoull(text, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer for --" + key + ": " + text);
    }
    if (text.empty() || text.front() == '-' || parsed != text.size()) {
        throw std::runtime_error("Invalid integer for --" + key + ": " + text);
    }
    return value;
}

inline int parseInteger(const std::string& key, const NextArgumentValue& next) {
    const std::string text = next();
    size_t            parsed{0};
    int               value{0};
    try {
        value = std::stoi(text, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer for --" + key + ": " + text);
    }
    if (parsed != text.size()) {
        throw std::runtime_error("Invalid integer for --" + key + ": " + text);
    }
    return value;
}

inline double parseDouble(const std::string& key, const NextArgumentValue& next) {
    const std::string text = next();
    size_t            parsed{0};
    double            value{0};
    try {
        value = std::stod(text, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid number for --" + key + ": " + text);
    }
    if (parsed != text.size()) {
        throw std::runtime_error("Invalid number for --" + key + ": " + text);
    }
    return value;
}

}  // namespace rtp_llm::benchmark
