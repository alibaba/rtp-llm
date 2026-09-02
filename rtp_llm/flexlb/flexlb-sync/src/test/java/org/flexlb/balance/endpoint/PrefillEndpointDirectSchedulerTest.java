package org.flexlb.balance.endpoint;

import org.flexlb.config.SchedulerConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.mock;

class PrefillEndpointDirectSchedulerTest {

    @Test
    void directSchedulerCanConstructPrefillEndpoint() {
        FlexlbConfig config = new FlexlbConfig();
        config.setScheduler(SchedulerConfig.direct());
        config.setDispatcher(DispatcherConfig.nonBatch());

        EndpointTestSupport.TestRequestRuntime requestRuntime =
                EndpointTestSupport.requestRuntime();
        PrefillEndpoint endpoint = assertDoesNotThrow(() -> {
            PrefillEndpoint created = new PrefillEndpoint(
                    workerStatus(),
                    config,
                    EndpointTestSupport.routeStrategy(requestRuntime),
                    requestRuntime,
                    requestRuntime,
                    mock(BatchSchedulerReporter.class));
            created.startGeneration();
            return created;
        });
        endpoint.close();
    }

    private static WorkerStatus workerStatus() {
        WorkerStatus status = EndpointTestSupport.workerStatus(
                RoleType.PREFILL, "127.0.0.81", 8081, 9081);
        return status;
    }
}
