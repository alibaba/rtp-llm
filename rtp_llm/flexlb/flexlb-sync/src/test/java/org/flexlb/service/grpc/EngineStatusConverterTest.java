package org.flexlb.service.grpc;

import org.flexlb.domain.worker.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EngineStatusConverterTest {

    @Test
    void leavesCacheStatusEmptyWhenWorkerDoesNotReportBlockSize() {
        EngineRpcService.WorkerStatusPB workerStatus =
                EngineRpcService.WorkerStatusPB.newBuilder().build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        assertNull(response.getCacheStatus());
    }

    @Test
    void preservesStringRequestIdFromWorkerStatus() {
        String requestId = "c68b72ff-982d-944f-9834-bc0e8bf2f43f";
        EngineRpcService.TaskInfoPB finishedTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(requestId)
                .build();
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .addFinishedTaskList(finishedTask)
                .build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        assertEquals(requestId, response.getFinishedTaskInfo().get(requestId).getRequestId());
    }

    @Test
    void preservesPrefixLengthValidityFromWorkerStatus() {
        EngineRpcService.TaskInfoPB runningTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId("request-1")
                .setPrefixLength(128)
                .setPrefixLengthValid(true)
                .build();
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .addRunningTaskInfo(runningTask)
                .build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        assertEquals(128, response.getRunningTaskInfo().get("request-1").getPrefixLength());
        assertTrue(response.getRunningTaskInfo().get("request-1").isPrefixLengthValid());
    }

    @Test
    void preservesBlockHashLookaheadTokensFromWorkerStatus() {
        EngineRpcService.WorkerStatusPB workerStatus = EngineRpcService.WorkerStatusPB.newBuilder()
                .setBlockHashLookaheadTokens(1)
                .build();

        WorkerStatusResponse response =
                EngineStatusConverter.convertToWorkerStatusResponse(workerStatus);

        assertEquals(1, response.getBlockHashLookaheadTokens());
    }
}
