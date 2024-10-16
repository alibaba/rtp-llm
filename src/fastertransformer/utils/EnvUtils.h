#pragma once

#include <string>

namespace fastertransformer {

inline std::string getEnvWithDefault(const std::string& name, const std::string& default_value) {
    if (std::getenv(name.c_str())) {
        return std::getenv(name.c_str());
    } else {
        return default_value;
    }
}

};