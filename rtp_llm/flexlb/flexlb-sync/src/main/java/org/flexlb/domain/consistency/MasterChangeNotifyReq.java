package org.flexlb.domain.consistency;

import lombok.Getter;
import lombok.Setter;

/**
 * Carries the master-change notification sent between FlexLB instances.
 */
@Setter
@Getter
public class MasterChangeNotifyReq {

    private String reqIp;
    private String roleId;
}
