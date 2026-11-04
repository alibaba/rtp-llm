#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace navi {
class AsyncPipe;
}

namespace navi_sched {

class MasterStateR;

class MasterDispatchHandle {
public:
    bool valid() const { return bool(_state); }
    bool registerWaiter(const std::shared_ptr<navi::AsyncPipe> &pipe,
                        bool &acknowledged) const;
    bool cancel(const std::shared_ptr<navi::AsyncPipe> &pipe) const;
    bool abandon() const;
    static void fail(const std::vector<MasterDispatchHandle> &handles);

private:
    friend class MasterStateR;
    struct State;
    std::shared_ptr<State> _state;
    size_t _requestIndex = 0;
};

}
