#pragma once

namespace autil {

class WorkItem {
public:
    WorkItem() {}
    virtual ~WorkItem() {}
    virtual void process() = 0;

public:
    virtual void destroy() { delete this; }
    virtual void drop() { destroy(); }
};

} // namespace autil
