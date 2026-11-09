package org.flexlb.httpserver;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * Runs receiving deadline, cancellation and control-admission regression scenarios.
 */
@Timeout(30)
class GrpcReceivingGuardTest {
    @Test
    void receivingDeadlineAndBusinessQueue() throws Exception {
        GrpcReceivingGuardRepro.main(new String[0]);
    }

    @Test
    void queuedProducerCancelDoesNotPrematurelyReleaseBudget() throws Exception {
        QueuedMessageCancelRepro.main(new String[0]);
    }

    @Test
    void rawProtocolControlAndTimerMatrix() throws Exception {
        GrpcReceivingGuardMatrix.main(new String[0]);
    }
}
