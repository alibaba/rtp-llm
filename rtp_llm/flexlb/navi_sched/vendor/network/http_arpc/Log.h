#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/alogadapter.h"
#include "alog/Configurator.h"
#include "alog/Logger.h"

#if !defined ALOG_VERSION || ALOG_VERSION < 010101
#error "From this version HTTP_ARPC requires ALOG >= 1.1.1. Please check your ALOG."
#endif

namespace http_arpc {

extern ILogger *httparpclogger;
inline ILogger *GetHttpArpcLogger() { return httparpclogger; }
inline void SetHttpArpcLogger(ILogger *l) { httparpclogger = l; }

#define HTTP_ARPC_LOG_CONFIG(config) (http_arpc::httparpclogger->logSetup(config))
#define HTTP_ARPC_LOG_SHUTDOWN() (http_arpc::httparpclogger->logTearDown())

/* ALOG_LOG will expand to logger->log(...), which complies with our ILogger
 * interface, so we just reuse ALOG_LOG for simplification. */
#define HTTP_ARPC_LOG(level, format, args...)                                                                          \
    ALOG_LOG(http_arpc::httparpclogger, alog::LOG_LEVEL_##level, format, ##args)

#define HTTP_ARPC_DECLARE_AND_SETUP_LOGGER(c) static alog::Logger *_logger = alog::Logger::getLogger("HTTPARPC." #c)

} // namespace http_arpc

