#pragma once

namespace anet {

/* Since we will leverage the SO_PRIORITY socket option, we will need
 * to map our priority to the real priority in linux kernel, that is
 * the output of "/sbin/tc -s qdisc", priority map section . */
typedef enum ConnectionPriority {
    ANET_PRIORITY_DEFAULT = 0,
    ANET_PRIORITY_LOW = 1,
    ANET_PRIORITY_NORMAL = 0,
    ANET_PRIORITY_HIGH = 6
} CONNPRIORITY;

inline bool isValidPriority(CONNPRIORITY prio) { return (prio >= ANET_PRIORITY_DEFAULT && prio <= ANET_PRIORITY_HIGH); }

}; // namespace anet
