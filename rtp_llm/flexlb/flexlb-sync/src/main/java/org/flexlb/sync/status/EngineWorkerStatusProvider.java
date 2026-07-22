package org.flexlb.sync.status;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Collection;

@Slf4j
@Service
public class EngineWorkerStatusProvider implements WorkerStatusProvider {
    
    @Autowired
    private EngineWorkerStatus engineWorkerStatus;

    @Override
    public Collection<WorkerStatus> getWorkerStatuses(RoleType roleType, String group) {
        return engineWorkerStatus.selectModelWorkerStatus(roleType, group).values();
    }
}
