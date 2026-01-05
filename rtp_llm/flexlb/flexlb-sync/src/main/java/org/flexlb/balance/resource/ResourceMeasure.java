package org.flexlb.balance.resource;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;

/**
 * 资源度量接口
 * 根据不同的RoleType,使用不同的资源可用性判断逻辑
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
public interface ResourceMeasure {

    /**
     * 检查指定worker资源是否可用
     *
     * @param workerStatus 单个worker状态
     * @return true表示该worker有可用资源,false表示无可用资源
     */
    boolean isResourceAvailable(WorkerStatus workerStatus);

    /**
     * 检查一组workers中是否至少有一个有可用资源
     *
     * @param modelName 模型名称
     * @param group     Worker分组(可为null,表示不限制分组)
     * @return true表示有可用Worker, false表示无可用Worker
     */
    boolean hasResourceAvailableWorker(String modelName, String group);

    /**
     * 获取资源评估指标
     *
     * @return 资源评估指标
     */
    ResourceMeasureIndicatorEnum getResourceMeasureIndicator();
}
