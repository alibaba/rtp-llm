package org.flexlb.master;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

/**
 * Master-change notification response.
 */
@ToString
@Getter
@Setter
public class MasterChangeNotifyResp {

    private boolean success;

    private String msg;
}
