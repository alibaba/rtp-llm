package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.PrefillActiveIndex;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.config.FlexlbConfig;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;

public final class PrefillQueueRunnerTestSupport {

    private PrefillQueueRunnerTestSupport() {}

    static CapacityBoundary boundaryOnly(RequestRoute item, CapacityBoundary boundary) {
        return boundary;
    }

    public static PrefillQueueRunner create(
            String key,
            PrefillEndpoint endpoint,
            FlexlbConfig config,
            DeliveryDispatcher delivery,
            AbstractRequestScheduler scheduler) {
        var runtime = new AtomicReference<PrefillQueueRunner>();
        var index =
                PrefillActiveIndex.ordered(
                        16,
                        config.isPriorityOrdering()
                                ? PrefillQueueRunner.PRIORITY_QUEUE_ORDER
                                : PrefillQueueRunner.FIFO_QUEUE_ORDER);
        var state = new PrefillState(new ReentrantLock(), index, System::currentTimeMillis);
        var worker =
                org.mockito.Mockito.mock(
                        PrefillQueueRunner.class,
                        org.mockito.Mockito.withSettings()
                                .useConstructor(
                                        key,
                                        endpoint,
                                        QueueExecutionSettings.capture(config),
                                        delivery,
                                        state)
                                .defaultAnswer(org.mockito.Mockito.CALLS_REAL_METHODS));
        org.mockito.Mockito.doAnswer(
                        call -> {
                            RequestRoute route = call.getArgument(0);
                            if (route.ctx().data().getScheduler() == null) {
                                SchedulerTestSupport.bindOwner(route.ctx(), scheduler);
                            }
                            return call.callRealMethod();
                        })
                .when(worker)
                .offer(org.mockito.ArgumentMatchers.any());
        org.springframework.test.util.ReflectionTestUtils.setField(endpoint, "queueRunner", worker);
        org.springframework.test.util.ReflectionTestUtils.setField(endpoint, "ownership", state);
        DeliveryTestSupport.installQueueFixture(worker, delivery);
        runtime.set(worker);
        return runtime.get();
    }

    public static PrefillState.QueueSnapshot capture(PrefillQueueRunner runtime) {
        return state(runtime).captureQueue(Integer.MAX_VALUE);
    }

    public static PrefillState state(PrefillQueueRunner runtime) {
        return (PrefillState) ReflectionTestUtils.getField(runtime, "prefillState");
    }
}
