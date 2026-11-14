package org.flexlb.domain.consistency;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

/**
 * Carries the result of a master-change notification.
 */
@ToString
@Getter
@Setter
public class MasterChangeNotifyResp {

    private boolean success;

    private String msg;
}
