package org.flexlb.domain.consistency;

import lombok.Getter;
import lombok.Setter;

/**
 * Carries the instance status submitted for FlexLB synchronization.
 */
@Setter
@Getter
public class SyncLBStatusReq {

    private String roleId;
}
