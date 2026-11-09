package org.flexlb.domain.consistency;

import lombok.Getter;
import lombok.Setter;

/**
 * Carries the response to a FlexLB status synchronization request.
 */
@Getter
@Setter
public class SyncLBStatusResp {

    private boolean success;
    private String msg;
    private String lbStatus;
}
