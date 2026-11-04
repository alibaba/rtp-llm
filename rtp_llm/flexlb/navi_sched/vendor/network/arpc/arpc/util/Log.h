#pragma once

#include <iostream>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/alogadapter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "alog/Configurator.h"
#include "alog/Logger.h"
#include "alog/Version.h"

#if !defined ALOG_VERSION || ALOG_VERSION < 010101
#error "From this version ARPC requires ALOG >= 1.1.1. Please check your ALOG."
#endif

ARPC_BEGIN_NAMESPACE(arpc);

extern ILogger *arpclogger;
inline ILogger *GetArpcLogger() { return arpclogger; }
inline void SetArpcLogger(ILogger *l) { arpclogger = l; }

#define ARPC_LOG_CONFIG(config) (arpc::arpclogger->logSetup(config))
#define ARPC_LOG_SHUTDOWN() (arpc::arpclogger->logTearDown())

/* ALOG_LOG will expand to logger->log(...), which complies with our ILogger
 * interface, so we just reuse ALOG_LOG for simplification. */
#define ARPC_LOG(level, format, args...) ALOG_LOG(arpc::arpclogger, alog::LOG_LEVEL_##level, format, ##args)
#define ARPC_DECLARE_AND_SETUP_LOGGER(c)

ARPC_END_NAMESPACE(arpc);

