package org.flexlb.master;

import lombok.Getter;
import lombok.Setter;

/**
 * Master-change notification identity.
 */
@Setter
@Getter
public class MasterChangeNotifyReq {

    private String reqIp;
    private String roleId;
}
