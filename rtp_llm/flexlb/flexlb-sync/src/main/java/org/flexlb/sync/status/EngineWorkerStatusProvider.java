package org.flexlb.sync.status;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Slf4j
@Service
public class EngineWorkerStatusProvider implements WorkerStatusProvider {

    @Autowired
    private EngineWorkerStatus engineWorkerStatus;

    @Override
    public List<String> getWorkerIpPorts(RoleType roleType, String group) {

        return engineWorkerStatus.modelWorkerAddresses(roleType, group);
    }
}
