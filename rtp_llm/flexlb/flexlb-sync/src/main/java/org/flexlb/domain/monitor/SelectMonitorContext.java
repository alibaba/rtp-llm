package org.flexlb.domain.monitor;

import lombok.Getter;
import lombok.Setter;

/**
 * @author zjw
 * description:
 * date: 2025/3/14
 */
@Getter
@Setter
public class SelectMonitorContext {

    private long startTime;

    private String errorCode;
    private long totalCost;

    private long tokenizeStartTime;
    private long tokenizeEndTime;

    private long calcPrefixStartTime;
    private long calcPrefixEndTime;

    private long calcTTFTStartTime;
    private long calcTTFTEndTime;

    public void markError(String errorCode) {
        this.errorCode = errorCode;
    }
}
