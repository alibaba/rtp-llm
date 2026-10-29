#pragma once

#include "aios/network/rdma/accl/RdmaMessage.h"
#include "aios/network/rdma/accl/RdmaRequest.h"

ARPC_BEGIN_NAMESPACE(arpc);

class FakeRdmaRequest: public RdmaRequest {
public:
    FakeRdmaRequest() {}

public:
    virtual std::shared_ptr<RdmaMessage> Encode() override;
    virtual bool                         Decode(const std::shared_ptr<RdmaMessage>& message) override;

    void SetValue(const std::string& value) {
        _value = value;
    }
    const std::string& GetValue() const {
        return _value;
    }

private:
    std::string _value;
};

ARPC_END_NAMESPACE(arpc);