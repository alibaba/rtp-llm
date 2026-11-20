package org.flexlb.master;

import lombok.Getter;
import lombok.Setter;

/**
 * Leadership snapshot query; wire fields are retained for compatibility.
 */
@Setter
@Getter
public class SyncLBStatusReq {

    private String roleId;
}
