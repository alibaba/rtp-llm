#pragma once

#include "kmonitor/client/core/MetricsRecord.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

class Sink {
public:
    Sink(const std::string &name) : name_(name) {}
    virtual ~Sink() {}
    virtual bool Init() = 0;
    virtual void PutMetrics(MetricsRecord *record) = 0;
    virtual void Flush() = 0;

public:
    const std::string GetName() const { return name_; }

private:
    std::string name_;
};

TYPEDEF_PTR(Sink);

END_KMONITOR_NAMESPACE(kmonitor);
