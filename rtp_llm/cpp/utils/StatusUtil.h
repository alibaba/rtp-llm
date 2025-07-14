#pragma once

#include "rtp_llm/cpp/utils/AssertUtils.h"

#define RETURN_IF_NOT_SUCCESS(success)                                                                                 \
    do {                                                                                                               \
        if (!(success)) {                                                                                              \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

#define RETURN_IF_STATUS_OR_ERROR(status_or)                                                                           \
    do {                                                                                                               \
        auto&& _status_or = (status_or);                                                                               \
        if (ABSL_PREDICT_FALSE(!_status_or.ok())) {                                                                    \
            RTP_LLM_LOG_ERROR("error msg: %s", _status_or.status().ToString().c_str());                                \
            return _status_or.status();                                                                                \
        }                                                                                                              \
    } while (0)

#define RETURN_IF_STATUS_ERROR(status)                                                                                 \
    do {                                                                                                               \
        auto _status = (status);                                                                                       \
        if (ABSL_PREDICT_FALSE(!_status.ok()))                                                                         \
            return _status;                                                                                            \
    } while (0)

#define THROW_IF_STATUS_ERROR(status)                                                                                  \
    do {                                                                                                               \
        ::absl::Status _status = (status);                                                                             \
        if (ABSL_PREDICT_FALSE(!_status.ok()))                                                                         \
            RTP_LLM_FAIL(_status.ToString());                                                                          \
    } while (0)

#define THROW_IF_STATUSOR_ERROR(status_or)                                                                             \
    do {                                                                                                               \
        ::absl::Status _status = (status_or.status());                                                                 \
        if (ABSL_PREDICT_FALSE(!_status.ok()))                                                                         \
            RTP_LLM_FAIL(_status.ToString());                                                                          \
    } while (0)

#define CHECK_AND_RETURN_REF(result_var, call)                                                                         \
    auto result_var##_status = (call);                                                                                 \
    RETURN_IF_STATUS_OR_ERROR(result_var##_status);                                                                    \
    auto& result_var = result_var##_status.value();

#define CHECK_AND_RETURN_CONST_REF(result_var, call)                                                                   \
    const auto result_var##_status = (call);                                                                           \
    RETURN_IF_STATUS_OR_ERROR(result_var##_status);                                                                    \
    const auto& result_var = result_var##_status.value();

#define CHECK_AND_ASSIGN(result_var, call)                                                                             \
    auto result_var##_status = (call);                                                                                 \
    RETURN_IF_STATUS_OR_ERROR(result_var##_status);                                                                    \
    result_var = result_var##_status.value();
