package org.flexlb.domain.consistency;

import lombok.Getter;
import lombok.Setter;

/**
 * @author zjw
 * description:
 * date: 2025/3/31
 */
@Getter
@Setter
public class SyncLBStatusResp {

    private boolean success;
    private String msg;
    private String lbStatus;

}
