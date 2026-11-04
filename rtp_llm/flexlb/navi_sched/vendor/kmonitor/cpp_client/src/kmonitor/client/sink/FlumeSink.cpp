/*
 * Copyright 2014-2020 Alibaba Inc. All rights reserved.
 * Created on: 2017-02-24 15:00
 * Author Name: xsank.mz
 * Author Email: xsank.mz@alibaba-inc.com
 */

#include "kmonitor/client/sink/FlumeSink.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "autil/Log.h"
#include "autil/StringUtil.h"
#include "kmonitor/client/common/Common.h"
#include "kmonitor/client/common/Util.h"
#include "kmonitor/client/core/MetricsTags.h"
#include "kmonitor/client/core/MetricsValue.h"
#include "kmonitor/client/metric/Metric.h"
#include "kmonitor/client/net/AgentClient.h"
#include "kmonitor/client/net/LogFileAgentClient.h"
#include "kmonitor/client/net/thrift/EventBuilder.h"
#include "kmonitor/client/net/thrift/ThriftFlumeEvent.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);
AUTIL_LOG_SETUP(kmonitor, FlumeSink);

using std::map;
using std::string;
using std::stringstream;
using std::vector;

const char *FlumeSink::NAME = "FlumeSink";
const string FlumeSink::AGENT_THRIFT_PORT("4141");
const std::string FlumeSink::HEADER_TENANT("tenant");

FlumeSink::FlumeSink(const std::string &tenant,
                     const std::string &addr,
                     const std::string &sockPath,
                     bool enableLogFileSink,
                     const MetricsNameMapping *nameMapping)
    : Sink(FlumeSink::NAME), tenant_(tenant), client_(nullptr), queue_(QUEUE_CAPACITY), nameMapping_(nameMapping) {
    started_ = false;
    remote_send_cnt_ = 0;

    std::string localAddr = addr;
    if (localAddr.empty()) {
        localAddr = getDefaultSinkAddr();
    }
    client_ = new AgentClient(localAddr, sockPath, SOCK_TIMEOUT_MS);

    event_buffer_.reset(new BatchFlumeEvent());

    if (enableLogFileSink) {
        AUTIL_LOG(INFO, "enable output metrics to log file");
        log_file_client_ = new LogFileAgentClient();
    } else {
        AUTIL_LOG(INFO, "disable output metrics to log file");
        log_file_client_ = nullptr;
    }
}

FlumeSink::FlumeSink(const std::string &tenant,
                     BaseAgentClient *client,
                     BaseAgentClient *logFileClient,
                     const MetricsNameMapping *nameMapping)
    : Sink(FlumeSink::NAME), tenant_(tenant), queue_(QUEUE_CAPACITY), nameMapping_(nameMapping) {
    started_ = false;
    remote_send_cnt_ = 0;
    client_ = client;
    event_buffer_.reset(new BatchFlumeEvent());
    log_file_client_ = logFileClient;
}

string FlumeSink::getDefaultSinkAddr() const { return Util::GetMachineName() + ":" + AGENT_THRIFT_PORT; }

FlumeSink::~FlumeSink() {
    if (client_ != nullptr) {
        delete client_;
        client_ = nullptr;
    }
    if (log_file_client_ != nullptr) {
        delete log_file_client_;
        log_file_client_ = nullptr;
    }
}

bool FlumeSink::Init() {
    if (!client_->Init()) {
        return false;
    }
    if (log_file_client_ != nullptr && !log_file_client_->Init()) {
        return false;
    }
    return true;
}

void FlumeSink::PutMetrics(MetricsRecord *record) {
    for (const auto &value : record->Values()) {
        // 总是汇报原始指标名
        addMetricEvent(value->Name(), record, value);

        // 如果有指标映射，同时汇报映射后的指标名
        if (nameMapping_ != nullptr) {
            const string &mappedName = nameMapping_->getMappedName(value->Name());
            // 只有当映射后的名称与原始名称不同时才汇报
            if (mappedName != value->Name()) {
                addMetricEvent(mappedName, record, value);
            }
        }
    }
}

void FlumeSink::addMetricEvent(const std::string &metricName, MetricsRecord *record, const MetricsValue *value) {
    const string &tagstring = record->Tags()->ToString();
    stringstream_ << metricName;
    stringstream_ << " " << record->Timestamp() << " " << value->Value() << " ";
    stringstream_ << tagstring;
    string eventBody = stringstream_.str();
    ThriftFlumeEvent *event = EventBuilder::Build(eventBody);
    stringstream_.str("");
    stringstream_.clear();
    event->AddHeader(FlumeSink::HEADER_TENANT, tenant_);
    if (value->Headers().size() > 0) {
        event->AddHeaders(value->Headers());
    }
    event_buffer_->AppendFlumeEvent(event);
}

void FlumeSink::Flush() {
    BatchFlumeEventPtr events = event_buffer_;
    if (!events->IsEmpty()) {
        AUTIL_LOG(DEBUG, "flume sink flush %lu metrics", events->Size());
        bool ret = client_->AppendBatch(events);
        if (!ret) {
            client_->Close();
        }
        if (log_file_client_ != nullptr) {
            ret = log_file_client_->AppendBatch(events);
            if (!ret) {
                log_file_client_->Close();
            }
        }
    }
    event_buffer_.reset(new BatchFlumeEvent());
}

END_KMONITOR_NAMESPACE(kmonitor);
