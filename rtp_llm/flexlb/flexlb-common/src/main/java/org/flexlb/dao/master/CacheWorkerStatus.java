package org.flexlb.dao.master;

import lombok.Data;

@Data
public class CacheWorkerStatus {

    private WorkerMetaInfo workerMetaInfo;

    private long expirationTime;
}
