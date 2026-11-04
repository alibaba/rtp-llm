#pragma once
/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2017-02-24 15:00
 * Author Name: xsank.mz
 * Author Email: xsank.mz@alibaba-inc.com
 */

#include <sstream>
#include <string>

#include "autil/CircularQueue.h"
#include "autil/Log.h"
#include "kmonitor/client/common/Common.h"
#include "kmonitor/client/core/MetricsConfig.h"
#include "kmonitor/client/core/MetricsRecord.h"
#include "kmonitor/client/net/BaseAgentClient.h"
#include "kmonitor/client/net/BatchFlumeEvent.h"
#include "kmonitor/client/net/thrift/ThriftFlumeEvent.h"
#include "kmonitor/client/sink/Sink.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

class AgentClient;

class FlumeSink : public Sink {
public:
    static const char *NAME;
    explicit FlumeSink(const std::string &tenant,
                       const std::string &addr,
                       const std::string &sockPath = "",
                       bool enableLogFileSink = false,
                       const MetricsNameMapping *nameMapping = nullptr);
    explicit FlumeSink(const std::string &tenant,
                       BaseAgentClient *client,
                       BaseAgentClient *logFileClient = nullptr,
                       const MetricsNameMapping *nameMapping = nullptr);
    virtual ~FlumeSink() override;
    virtual bool Init() override;
    virtual void PutMetrics(MetricsRecord *record) override;
    virtual void Flush() override;
    bool ReConnect();

private:
    FlumeSink(const FlumeSink &);
    FlumeSink &operator=(const FlumeSink &);
    std::string getDefaultSinkAddr() const;
    void addMetricEvent(const std::string &metricName, MetricsRecord *record, const MetricsValue *value);

private:
    bool started_;
    std::string tenant_;
    BaseAgentClient *client_;
    BaseAgentClient *log_file_client_ = nullptr;
    BatchFlumeEventPtr event_buffer_;
    int remote_send_cnt_;
    autil::CircularQueue<BatchFlumeEventPtr> queue_;
    std::stringstream stringstream_;

private:
    static const int32_t SOCK_TIMEOUT_MS = 100;
    static const size_t QUEUE_CAPACITY = 100;
    static const int CLOSE_REMOTE_FREQ = 20;
    static const int REMOTE_SEND_TIMES = 3;

    static const std::string AGENT_THRIFT_PORT;
    static const std::string HEADER_TENANT;
    const MetricsNameMapping *nameMapping_;

private:
    AUTIL_LOG_DECLARE();

private:
    friend class FlumeSinkTest_TestLogFileOption_Test;
};

END_KMONITOR_NAMESPACE(kmonitor);
