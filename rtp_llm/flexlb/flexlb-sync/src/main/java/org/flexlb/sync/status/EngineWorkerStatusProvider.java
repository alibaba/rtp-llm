package org.flexlb.sync.status;

import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class EngineWorkerStatusProvider implements WorkerStatusProvider {

    private final EngineWorkerStatus engineWorkerStatus;

    public EngineWorkerStatusProvider(EngineWorkerStatus engineWorkerStatus) {
        this.engineWorkerStatus = engineWorkerStatus;
    }

    @Override
    public List<String> getWorkerIpPorts(RoleType roleType, String group) {
        return engineWorkerStatus.modelWorkerAddresses(roleType, group);
    }
}
