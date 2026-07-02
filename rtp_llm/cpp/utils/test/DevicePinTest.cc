#include "gtest/gtest.h"

#include "rtp_llm/cpp/utils/DevicePin.h"

#include <stdexcept>
#include <vector>

namespace rtp_llm {

TEST(DevicePinTest, NegativeDeviceIdSkipsSetterAndKeepsCache) {
    int cached_device = 3;
    int set_calls     = 0;

    detail::setCurrentThreadDeviceIfNeededImpl(-1, cached_device, [&set_calls](int) { ++set_calls; });

    ASSERT_EQ(0, set_calls);
    ASSERT_EQ(3, cached_device);
}

TEST(DevicePinTest, CachedDeviceIdSkipsRepeatedSetDevice) {
    int              cached_device = -1;
    std::vector<int> set_devices;

    auto set_device = [&set_devices](int device) { set_devices.push_back(device); };

    detail::setCurrentThreadDeviceIfNeededImpl(0, cached_device, set_device);
    detail::setCurrentThreadDeviceIfNeededImpl(0, cached_device, set_device);

    ASSERT_EQ(1u, set_devices.size());
    ASSERT_EQ(0, set_devices[0]);
    ASSERT_EQ(0, cached_device);
}

TEST(DevicePinTest, DifferentDeviceIdRetargetsThread) {
    int              cached_device = 0;
    std::vector<int> set_devices;

    detail::setCurrentThreadDeviceIfNeededImpl(
        1, cached_device, [&set_devices](int device) { set_devices.push_back(device); });

    ASSERT_EQ(1u, set_devices.size());
    ASSERT_EQ(1, set_devices[0]);
    ASSERT_EQ(1, cached_device);
}

TEST(DevicePinTest, SetterExceptionPropagatesAndDoesNotUpdateCache) {
    int cached_device = 0;

    ASSERT_THROW(detail::setCurrentThreadDeviceIfNeededImpl(
                     1, cached_device, [](int) { throw std::runtime_error("set device failed"); }),
                 std::runtime_error);
    ASSERT_EQ(0, cached_device);

    int set_calls = 0;
    detail::setCurrentThreadDeviceIfNeededImpl(1, cached_device, [&set_calls](int) { ++set_calls; });

    ASSERT_EQ(1, set_calls);
    ASSERT_EQ(1, cached_device);
}

}  // namespace rtp_llm
