package org.flexlb.service.grpc;

import org.flexlb.domain.worker.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNull;

class EngineStatusConverterTest {

    @Test
    void leavesCacheStatusEmptyWhenWorkerDoesNotReportBlockSize() {
        EngineRpcService.WorkerStatusPB workerStatus =
                EngineRpcService.WorkerStatusPB.newBuilder().build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        assertNull(response.getCacheStatus());
    }
}
