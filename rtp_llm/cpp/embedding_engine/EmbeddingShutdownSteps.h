#pragma once

#include <exception>
#include <functional>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// Runs shutdown steps independently: a failing step is logged and stays pending for a later
// call, the remaining steps still run, and the first failure is rethrown afterwards.
class EmbeddingShutdownSteps {
public:
    void run(bool& done, const char* name, const std::function<void()>& step) {
        if (done) {
            return;
        }
        try {
            step();
            done = true;
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("embedding server stop: %s failed: %s", name, e.what());
            keep(std::current_exception());
        } catch (...) {
            RTP_LLM_LOG_ERROR("embedding server stop: %s failed with unknown exception", name);
            keep(std::current_exception());
        }
    }

    void rethrowFirstError() const {
        if (first_error_) {
            std::rethrow_exception(first_error_);
        }
    }

private:
    void keep(std::exception_ptr error) {
        if (!first_error_) {
            first_error_ = std::move(error);
        }
    }

    std::exception_ptr first_error_;
};

}  // namespace rtp_llm
