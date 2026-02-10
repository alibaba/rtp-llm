package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;

import java.util.List;

/**
 * Worker状态提供者接口
 * 
 * @author FlexLB
 */
public interface WorkerStatusProvider {

    /**
     * 获取所有Worker的IP:Port
     *
     * @param roleType  查询的引擎角色
     * @param group     查询的引擎组
     * @return Worker IP:Port列表
     */
    List<String> getWorkerIpPorts(RoleType roleType, String group);
}