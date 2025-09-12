package org.flexlb.dao.master;

import java.util.List;

import org.flexlb.dao.route.RoleType;

/**
 * Worker状态提供者接口
 * 
 * @author FlexLB
 */
public interface WorkerStatusProvider {

    /**
     * 获取所有Worker的IP:Port
     *
     * @param modelName 模型名称
     * @param roleType  查询的引擎角色
     * @param group     查询的引擎组
     * @return Worker IP:Port列表
     */
    List<String> getWorkerIpPorts(String modelName, RoleType roleType, String group);
}