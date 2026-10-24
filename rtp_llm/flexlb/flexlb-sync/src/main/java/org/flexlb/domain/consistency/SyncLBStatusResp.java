package org.flexlb.domain.consistency;

import lombok.Getter;
import lombok.Setter;

/** Leadership snapshot response. */
@Getter
@Setter
public class SyncLBStatusResp {

    private boolean success;
    private String msg;
    private String lbStatus;

}
