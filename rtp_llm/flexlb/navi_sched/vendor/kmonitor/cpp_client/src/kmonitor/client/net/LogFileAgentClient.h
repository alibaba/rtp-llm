#pragma once

#include <string>
#include <vector>

#include "autil/Log.h"
#include "kmonitor/client/common/Common.h"
#include "kmonitor/client/net/BaseAgentClient.h"
#include "kmonitor/client/net/BatchFlumeEvent.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

class LogFileAgentClient : public BaseAgentClient {
public:
    LogFileAgentClient();
    ~LogFileAgentClient();

    bool AppendBatch(const BatchFlumeEventPtr &events) override;
    bool Started() const override;
    bool Init() override;
    void Close() override;
    bool ReConnect() override;

public:
    void AppendBatch(const std::vector<std::pair<std::string, std::string>> &records);

private:
    bool started_ = false;

private:
    AUTIL_LOG_DECLARE();
};

END_KMONITOR_NAMESPACE(kmonitor);
