package org.flexlb.dao.master;

import lombok.Data;
import org.flexlb.enums.BalanceStatusEnum;

/**
 * 获取模型all worker info 信息上下文
 */
@Data
public class ModelWorkerInfoServiceContext {

    private String modelName;

    private boolean success = true;

    private BalanceStatusEnum statusEnum = BalanceStatusEnum.SUCCESS;

    private String errorMsg;

    private long startTimeInMs = System.currentTimeMillis();

    private long totalRt;
}
