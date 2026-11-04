#include "rtp_llm/flexlb/navi_sched/service/NaviSchedRole.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "unittest/unittest.h"

#include <cstdlib>

namespace navi_sched {

class KmonExitTestRole : public NaviSchedRole {
public:
    explicit KmonExitTestRole(const NaviSchedConfig &config) : NaviSchedRole(config) {}
};

void verifyKmonStoppedAtExit()
{
    if (kmonitor::KMonitorFactory::IsStarted()) {
        std::_Exit(EXIT_FAILURE);
    }
}

TEST(KmonExitTest, testOwnedKmonShutsDownDuringProcessExit)
{
    kmonitor::KMonitorFactory::Shutdown();
    ASSERT_EQ(0, std::atexit(verifyKmonStoppedAtExit));
    NaviSchedConfig config;
    config.name = "exit_kmon_role";
    config.logFileName.clear();
    config.enableKmon = true;
    config.enableSnapshot = false;

    KmonExitTestRole role(config);
    ASSERT_TRUE(role.initKmon());
    ASSERT_TRUE(kmonitor::KMonitorFactory::IsStarted());

    role.stop();
    ASSERT_TRUE(kmonitor::KMonitorFactory::IsStarted());
}

} // namespace navi_sched
