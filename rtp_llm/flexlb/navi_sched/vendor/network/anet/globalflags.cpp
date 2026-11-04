#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/globalflags.h"

#include <sstream>
#include <stdint.h>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/crc.h"

namespace anet {
namespace flags {

#define DEFINE_GLOBAL_FLAG(type, name, defaultvalue)                                                                   \
    static type name = defaultvalue;                                                                                   \
    type get##name(void) { return name; }                                                                              \
    void set##name(type newval) { name = newval; }

/*************************************************************************/
DEFINE_GLOBAL_FLAG(int, ConnectTimeout, 5000000);
DEFINE_GLOBAL_FLAG(bool, ChecksumState, true);
DEFINE_GLOBAL_FLAG(int64_t, MaxConnectionCount, -1);
DEFINE_GLOBAL_FLAG(bool, EnableSocketNonBlockConnect, true);

/* Dump all global config for debugging purpose. */
int dump(std::ostringstream &buf) {
    buf << "=================================Dump of Config=================================\n";
    buf << "Checksum enabled: " << getChecksumState() << "\t"
        << "algorithm: " << std::string(GetCrcFuncStr()) << std::endl;

    buf << "Connecting Timout(us): " << getConnectTimeout() << std::endl;
    return 0;
}
} // namespace flags
} // namespace anet
