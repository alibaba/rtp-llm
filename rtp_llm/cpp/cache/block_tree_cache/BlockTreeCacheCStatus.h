#pragma once

#include "block_tree_cache.h"

#include <filesystem>
#include <ios>
#include <stdexcept>

namespace rtp_llm::c_api {
btc_status rememberError(btc_status status, const char* message);

template<typename F>
btc_status status(F&& f) {
    try {
        return f();
    } catch (const std::invalid_argument& e) {
        return rememberError(BTC_INVALID, e.what());
    } catch (const std::out_of_range& e) {
        return rememberError(BTC_INVALID, e.what());
    } catch (const std::ios_base::failure& e) {
        return rememberError(BTC_IO, e.what());
    } catch (const std::filesystem::filesystem_error& e) {
        return rememberError(BTC_IO, e.what());
    } catch (const std::exception& e) {
        return rememberError(BTC_INTERNAL, e.what());
    } catch (...) {
        return rememberError(BTC_INTERNAL, "Unknown native BlockTreeCache failure");
    }
}
template<typename T>
T& require(T* value) {
    if (!value)
        throw std::invalid_argument("Missing BlockTreeCache argument");
    return *value;
}
}  // namespace rtp_llm::c_api
