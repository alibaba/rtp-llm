package org.flexlb.domain.balance;

import lombok.Data;
import org.flexlb.dao.RequestContext;
import org.flexlb.dao.loadbalance.MasterRequest;
import org.flexlb.dao.loadbalance.MasterResponse;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.pv.PvLogData;

/**
 * @author zjw
 * description:
 * date: 2025/3/11
 */
@Data
public class BalanceContext {

    private WhaleMasterConfig config;

    private MasterRequest masterRequest;

    private MasterResponse masterResponse;

    private RequestContext requestContext;

    private ServerStatus serverStatus;

    private long interRequestId;

    private long startTime = System.currentTimeMillis();

    private int workerCalcParallel = 1;

    private boolean success = true;

    private PvLogData pvLogData;

}
