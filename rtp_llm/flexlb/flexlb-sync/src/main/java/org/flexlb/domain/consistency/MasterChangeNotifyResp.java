package org.flexlb.domain.consistency;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

/**
 * @author zjw
 * description:
 * date: 2025/3/31
 */
@ToString
@Getter
@Setter
public class MasterChangeNotifyResp {

    private boolean success;

    private String msg;

}
