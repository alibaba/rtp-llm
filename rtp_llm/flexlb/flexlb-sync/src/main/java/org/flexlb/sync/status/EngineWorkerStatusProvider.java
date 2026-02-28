package org.flexlb.sync.status;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
public class EngineWorkerStatusProvider implements WorkerStatusProvider {
    
    @Autowired
    private EngineWorkerStatus engineWorkerStatus;
    
    @Override
    public List<String> getWorkerIpPorts(RoleType roleType, String group) {

        Map<String/*ip:port*/, WorkerStatus> workerStatusMap
                = engineWorkerStatus.selectModelWorkerStatus(roleType, group);

        return new ArrayList<>(workerStatusMap.keySet());
    }
}