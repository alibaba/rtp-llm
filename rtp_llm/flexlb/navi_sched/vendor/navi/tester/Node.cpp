#include "navi/engine/Node.h"

#include "navi/engine/test/MockAsyncPipe.h"
#include "navi/tester/NaviResDummyKernel.h"

namespace navi {

AsyncPipePtr __attribute__((weak)) Node::createAsyncPipe(ActivateStrategy activateStrategy) {
    if (_runState && dynamic_cast<NaviResDummyKernel *>(_runState->kernel) != nullptr) {
        return std::make_shared<MockAsyncPipe>();
    } else {
        return doCreateAsyncPipe(activateStrategy);
    }
}

} // namespace navi
