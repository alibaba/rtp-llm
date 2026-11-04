#include "kmonitor/client/net/LogFileAgentClient.h"

#include <dirent.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "autil/Log.h"
#include "autil/StringUtil.h"
#include "kmonitor/client/common/Common.h"
#include "kmonitor/client/common/Util.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);
AUTIL_LOG_SETUP(kmonmetrics, LogFileAgentClient);

using namespace std;

LogFileAgentClient::LogFileAgentClient() {}

LogFileAgentClient::~LogFileAgentClient() { Close(); }

bool LogFileAgentClient::Init() { return true; }

bool LogFileAgentClient::Started() const { return started_; }

void LogFileAgentClient::Close() { started_ = false; }

bool LogFileAgentClient::AppendBatch(const BatchFlumeEventPtr &batchEvents) {
    if (!Started()) {
        ReConnect();
    }
    vector<ThriftFlumeEvent *> events = *(batchEvents->Get());
    for (auto event : events) {
        string eventBody = event->GetBody();
        AUTIL_LOG(INFO, "%s", eventBody.c_str());
    }
    return true;
}
void LogFileAgentClient::AppendBatch(const vector<pair<string, string>> &records) {
    for (auto &record : records) {
        AUTIL_LOG(INFO, "%s %s", record.first.c_str(), record.second.c_str());
    }
}

bool LogFileAgentClient::ReConnect() {
    Close();
    started_ = true;
    return true;
}

END_KMONITOR_NAMESPACE(kmonitor);
