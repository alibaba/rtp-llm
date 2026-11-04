#pragma once

#include <functional>

#include "autil/WorkItem.h"

namespace autil {

class LambdaWorkItem : public autil::WorkItem {
public:
    LambdaWorkItem(std::function<void()> func) : _func(std::move(func)) {}

public:
    void process() override {
        if (_func) {
            _func();
        }
    }

private:
    std::function<void()> _func;
};

} // namespace autil
