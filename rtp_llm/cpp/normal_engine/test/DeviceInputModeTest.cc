#include "rtp_llm/cpp/normal_engine/DeviceInputMode.h"
#include <gtest/gtest.h>
#include <array>
#include <string>

namespace rtp_llm {
namespace {

class ScopedEnv {
public:
    explicit ScopedEnv(const char* name): name_(name), was_set_(std::getenv(name) != nullptr) {
        if (was_set_) {
            value_ = std::getenv(name);
        }
    }
    ~ScopedEnv() {
        set(was_set_ ? value_.c_str() : nullptr);
    }
    void set(const char* value) {
        if (value) {
            setenv(name_, value, 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    bool        was_set_;
    std::string value_;
};

TEST(DeviceInputModeTest, AsyncOverridesMissingOrDisabledDeviceInput) {
    ScopedEnv stream_async("RTP_LLM_STREAM_ASYNC");
    ScopedEnv device_input("RTP_LLM_DEVICE_INPUT");
    stream_async.set("1");
    for (const char* value : std::array<const char*, 5>{nullptr, "0", "1", "", "true"}) {
        SCOPED_TRACE(value ? value : "(unset)");
        device_input.set(value);
        EXPECT_TRUE(deviceInputEnabled());
    }
}

TEST(DeviceInputModeTest, SyncPreservesExplicitDeviceInputChoice) {
    ScopedEnv stream_async("RTP_LLM_STREAM_ASYNC");
    ScopedEnv device_input("RTP_LLM_DEVICE_INPUT");
    for (const char* value : std::array<const char*, 4>{nullptr, "0", "", "true"}) {
        SCOPED_TRACE(value ? value : "(unset)");
        stream_async.set(value);
        device_input.set(nullptr);
        EXPECT_FALSE(deviceInputEnabled());
        device_input.set("0");
        EXPECT_FALSE(deviceInputEnabled());
        device_input.set("1");
        EXPECT_TRUE(deviceInputEnabled());
    }
}

}  // namespace
}  // namespace rtp_llm
