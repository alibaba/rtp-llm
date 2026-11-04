#include "autil/Regex.h"

#include <algorithm>
#include <regex.h>
#include <stdint.h>

#include "autil/Log.h"

#define SUBSLEN 64

namespace autil {

AUTIL_DECLARE_AND_SETUP_LOGGER(autil, Regex);

Regex::Regex() {}

Regex::~Regex() {}

bool Regex::match(const std::string &str, const std::string &pattern) {
    return match(str.c_str(), pattern, REG_EXTENDED);
};

bool Regex::match(const std::string &str, const std::string &pattern, int regex_mode) {
    return match(str.c_str(), pattern, regex_mode);
};

bool Regex::match(const char *str, const std::string &pattern) { return match(str, pattern, REG_EXTENDED); }

bool Regex::match(const char *str, const std::string &pattern, int regex_mode) {
    regex_t reg;
    int rtn;

    rtn = regcomp(&reg, pattern.c_str(), regex_mode);
    if (rtn != 0) {
        AUTIL_LOG(ERROR, "Invalid regex pattern: %s", pattern.c_str());
        return false;
    }
    rtn = regexec(&reg, str, 0, 0, 0);
    regfree(&reg);
    switch (rtn) {
    case 0:
        return true;
    case REG_NOMATCH:
        return false;
    default:
        AUTIL_LOG(ERROR, "Error when matching regex: %s with: %s", str, pattern.c_str());
        return false;
    }
}

bool Regex::groupMatch(const std::string &str, const std::string &pattern, std::vector<std::string> &matched_items) {
    bool IsFind;
    // size_t len;
    regex_t regex;
    int err;
    regmatch_t subs[SUBSLEN];

    err = regcomp(&regex, pattern.c_str(), REG_EXTENDED);
    if (err != 0) {
        regfree(&regex);
        AUTIL_LOG(ERROR, "Invalid regex pattern: %s", pattern.c_str());
        return false;
    } else {
        err = regexec(&regex, str.c_str(), (size_t)SUBSLEN, subs, 0);
    }

    switch (err) {
    case 0:
        IsFind = true;
        break;
    case REG_NOMATCH:
        IsFind = false;
        break;
    default:
        regfree(&regex);
        AUTIL_LOG(ERROR, "Error when matching regex: %s with: %s", str.c_str(), pattern.c_str());
        return false;
    }
    if (IsFind == false) {
        regfree(&regex);
        return false;
    }

    matched_items.clear();
    for (decltype(regex.re_nsub) i = 0; i <= regex.re_nsub; i++) {
        int32_t length = subs[i].rm_eo - subs[i].rm_so;
        if (i == 0) {
            continue; /// the first match is the whole string
        } else {
            matched_items.push_back(std::string(str.c_str() + subs[i].rm_so, length));
        }
    }
    regfree(&regex);
    return IsFind;
}

} // namespace autil
