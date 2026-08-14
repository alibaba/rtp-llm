#include "gtest/gtest.h"

#include "rtp_llm/cpp/models/PyWrappedModel.h"

#include <memory>

namespace py = pybind11;

namespace rtp_llm {
namespace {

class FakeEvent {
public:
    struct State {
        bool completed         = false;
        int  query_count       = 0;
        int  synchronize_count = 0;
    };

    explicit FakeEvent(std::shared_ptr<State> state): state_(std::move(state)) {}

    bool query() const {
        ++state_->query_count;
        return state_->completed;
    }

    void synchronize() const {
        ++state_->synchronize_count;
        state_->completed = true;
    }

private:
    std::shared_ptr<State> state_;
};

class AttnPyObjectRetirementQueueTest: public ::testing::Test {
protected:
    using Queue = PyWrappedModel::AttnPyObjectRetirementQueue<FakeEvent>;

    static void SetUpTestSuite() {
        interpreter_ = std::make_unique<py::scoped_interpreter>();
    }

    static void TearDownTestSuite() {
        interpreter_.reset();
    }

    static std::pair<py::object, py::object> makeTrackedObject(py::list destroyed) {
        py::dict globals;
        globals["__builtins__"] = py::module_::import("builtins");
        globals["destroyed"]    = destroyed;
        py::exec(R"(
class Tracker:
    def __del__(self):
        destroyed.append("released")
)",
                 globals);
        py::object object = globals["Tracker"]();
        py::object weak   = py::module_::import("weakref").attr("ref")(object);
        return {std::move(object), std::move(weak)};
    }

private:
    static std::unique_ptr<py::scoped_interpreter> interpreter_;
};

std::unique_ptr<py::scoped_interpreter> AttnPyObjectRetirementQueueTest::interpreter_;

TEST_F(AttnPyObjectRetirementQueueTest, PendingEventKeepsOldObjectAliveUntilCompletion) {
    Queue    queue;
    py::list destroyed;
    auto     tracked = makeTrackedObject(destroyed);
    auto     state   = std::make_shared<FakeEvent::State>();
    auto     event   = std::make_shared<FakeEvent>(state);

    queue.retire(std::move(tracked.first), event);
    ASSERT_EQ(queue.size(), 1);

    // Models a second normal forward replacing held_attn_pyobj_ while the first
    // forward's planner DMA/kernel work is still pending.
    queue.releaseCompleted();
    EXPECT_EQ(queue.size(), 1);
    EXPECT_FALSE(tracked.second().is_none());
    EXPECT_EQ(py::len(destroyed), 0);

    state->completed = true;
    queue.releaseCompleted();
    EXPECT_TRUE(queue.empty());
    EXPECT_TRUE(tracked.second().is_none());
    EXPECT_EQ(py::len(destroyed), 1);
    EXPECT_GE(state->query_count, 2);
}

TEST_F(AttnPyObjectRetirementQueueTest, ShutdownSynchronizesPendingEventBeforeRelease) {
    Queue    queue;
    py::list destroyed;
    auto     tracked = makeTrackedObject(destroyed);
    auto     state   = std::make_shared<FakeEvent::State>();
    auto     event   = std::make_shared<FakeEvent>(state);

    queue.retire(std::move(tracked.first), event);
    queue.synchronizeAndReleaseAll();

    EXPECT_TRUE(queue.empty());
    EXPECT_TRUE(state->completed);
    EXPECT_EQ(state->synchronize_count, 1);
    EXPECT_TRUE(tracked.second().is_none());
    EXPECT_EQ(py::len(destroyed), 1);
}

}  // namespace
}  // namespace rtp_llm
