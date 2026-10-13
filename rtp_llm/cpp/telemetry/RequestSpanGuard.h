#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <utility>

#include "opentelemetry/common/timestamp.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/trace/span.h"
#include "opentelemetry/trace/span_metadata.h"

namespace rtp_llm {
namespace telemetry {

// Idempotent RAII finish guard for an OTel span.
// - finish() is exactly-once via atomic flag; we do NOT rely on the SDK
//   tolerating double End(), which is not a documented guarantee.
// - Destructor is a noexcept fallback that swallows everything (fail-open):
//   CHECK_ERROR_STATUS / EXECUTE_STAGE_FUNC early returns and exceptions all
//   end the span through stack unwinding.
// - Terminal attributes/status must be written before End(); use span() then
//   finish(), or the convenience finish(status, description).
class RequestSpanGuard {
public:
    RequestSpanGuard() = default;
    explicit RequestSpanGuard(opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> span):
        span_(std::move(span)) {}

    RequestSpanGuard(const RequestSpanGuard&)            = delete;
    RequestSpanGuard& operator=(const RequestSpanGuard&) = delete;

    ~RequestSpanGuard() noexcept {
        try {
            finish();
        } catch (...) {
            // telemetry must never throw out of a destructor
        }
    }

    bool valid() const {
        return span_ != nullptr;
    }

    // Shared handle for establishing parent/child relations across calls.
    opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> sharedSpan() const {
        return span_;
    }

    void setAttribute(opentelemetry::nostd::string_view            key,
                      const opentelemetry::common::AttributeValue& value) noexcept {
        try {
            if (span_ && !finished_.load(std::memory_order_acquire)) {
                span_->SetAttribute(key, value);
            }
        } catch (...) {}
    }

    // Span event with an explicit epoch-µs timestamp (post-hoc marking, same
    // technique as PhaseSpanSynthesizer's start_system_time). Dropped after
    // finish(); fail-open.
    void addEvent(opentelemetry::nostd::string_view name, int64_t epoch_us) noexcept {
        try {
            if (span_ && !finished_.load(std::memory_order_acquire)) {
                span_->AddEvent(name, opentelemetry::common::SystemTimestamp(std::chrono::microseconds(epoch_us)));
            }
        } catch (...) {}
    }

    // Exactly-once End(); later calls are no-ops.
    void finish() noexcept {
        try {
            if (!span_ || finished_.exchange(true, std::memory_order_acq_rel)) {
                return;
            }
            span_->End();
        } catch (...) {}
    }

    // Sets status (+ optional description) then ends the span.
    void finish(opentelemetry::trace::StatusCode status, const std::string& description = "") noexcept {
        try {
            if (!span_ || finished_.load(std::memory_order_acquire)) {
                return;
            }
            span_->SetStatus(status, description);
        } catch (...) {}
        finish();
    }

private:
    opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> span_;
    std::atomic<bool>                                            finished_{false};
};

}  // namespace telemetry
}  // namespace rtp_llm
