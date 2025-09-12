package org.flexlb.domain.consistency;

import lombok.Getter;
import lombok.Setter;

/**
 * @author zjw
 * description:
 * date: 2025/3/31
 */
@Setter
@Getter
public class MasterChangeNotifyReq {

    private String reqIp;
    private String roleId;

}
