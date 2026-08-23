package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.AdmittedDecisionGroup;
import org.flexlb.balance.scheduler.DecisionGroupHandler;
import org.flexlb.balance.scheduler.DecisionGroupMetadata;
import org.flexlb.balance.scheduler.TestCapacityAdmission;
import org.flexlb.config.DirectSchedulerConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.mock;

class PrefillEndpointDirectSchedulerTest {

    @Test
    void directSchedulerCanConstructPrefillEndpoint() {
        FlexlbConfig config = new FlexlbConfig();
        config.setScheduler(new DirectSchedulerConfig());
        config.setDispatcher(new NonBatchDispatcherConfig());

        PrefillEndpoint endpoint = assertDoesNotThrow(() -> new PrefillEndpoint(
                workerStatus(), config, noopHandler(),
                TestCapacityAdmission.alwaysAvailable(),
                mock(BatchSchedulerReporter.class)));
        endpoint.close();
    }

    private static WorkerStatus workerStatus() {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.81");
        status.setPort(8081);
        status.setGrpcPort(9081);
        status.setRole(RoleType.PREFILL);
        return status;
    }

    private static DecisionGroupHandler noopHandler() {
        return new DecisionGroupHandler() {
            @Override public void onExpired(BatchItem head) { }
            @Override public void onDecisionGroupAdmitted(
                    AdmittedDecisionGroup group, DecisionGroupMetadata metadata) {
                TestCapacityAdmission.complete(group);
            }
            @Override public void onOfferFailure(BatchItem item, Throwable error) { }
            @Override public void onDeliveryFailure(BatchItem item, Throwable error) { }
        };
    }
}
