package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint.CapacityUsage;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRequestView;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRoutingView;
import org.flexlb.balance.endpoint.DecodeEndpoint.LayeredAdmissionView;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.enums.DecodeTaskPhase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

class RouteAdmissionCapacityTest {

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void prefillUsesCurrentAdmissionCapacityAndClosesThePin(boolean canAccept) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        WorkerEndpoint.GenerationPin pin = mock(WorkerEndpoint.GenerationPin.class);
        when(endpoint.tryPinGeneration()).thenReturn(pin);
        when(endpoint.canAcceptRequest()).thenReturn(canAccept);
        BalanceContext context = context(1L, 2L, 90L, 5L, 5L);
        SchedulingTestConfig.disallowVictim(context.getConfig(), VictimStage.PREFILL_QUEUED);

        assertEquals(!canAccept,
                RouteAdmission.mustWaitForCapacity(context, endpoint));

        verify(endpoint).canAcceptRequest();
        verify(pin).close();
    }

    @Test
    void fullPrefillWithAllowedQueuedVictimsLeavesReclamationToFreshPlacement() {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        WorkerEndpoint.GenerationPin pin = mock(WorkerEndpoint.GenerationPin.class);
        when(endpoint.tryPinGeneration()).thenReturn(pin);
        when(endpoint.canAcceptRequest()).thenReturn(false);
        BalanceContext context = context(1L, 2L, 90L, 5L, 5L);
        SchedulingTestConfig.allowVictim(context.getConfig(), VictimStage.PREFILL_QUEUED);
        when(endpoint.canPreemptQueuedRequest(context.getPriority())).thenReturn(true);

        assertFalse(RouteAdmission.mustWaitForCapacity(context, endpoint));

        verify(endpoint).canPreemptQueuedRequest(context.getPriority());
        verify(pin).close();
    }

    @Test
    void fullPrefillStillWaitsWhenQueuedPreemptionIsDisabled() {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        WorkerEndpoint.GenerationPin pin = mock(WorkerEndpoint.GenerationPin.class);
        when(endpoint.tryPinGeneration()).thenReturn(pin);
        when(endpoint.canAcceptRequest()).thenReturn(false);
        BalanceContext context = context(1L, 2L, 90L, 5L, 5L);
        SchedulingTestConfig.disallowVictim(context.getConfig(), VictimStage.PREFILL_QUEUED);
        when(endpoint.canPreemptQueuedRequest(context.getPriority())).thenReturn(true);

        assertTrue(RouteAdmission.mustWaitForCapacity(context, endpoint));

        verify(endpoint, never()).canPreemptQueuedRequest(context.getPriority());
        verify(pin).close();
    }

    @Test
    void fullPrefillStillWaitsWhenNoQueuedVictimCanBePreempted() {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        WorkerEndpoint.GenerationPin pin = mock(WorkerEndpoint.GenerationPin.class);
        when(endpoint.tryPinGeneration()).thenReturn(pin);
        when(endpoint.canAcceptRequest()).thenReturn(false);
        BalanceContext context = context(1L, 2L, 90L, 5L, 5L);
        SchedulingTestConfig.allowVictim(context.getConfig(), VictimStage.PREFILL_QUEUED);
        when(endpoint.canPreemptQueuedRequest(context.getPriority())).thenReturn(false);

        assertTrue(RouteAdmission.mustWaitForCapacity(context, endpoint));

        verify(endpoint).canPreemptQueuedRequest(context.getPriority());
        verify(pin).close();
    }

    @Test
    void unavailableGenerationIsUnknownWithoutReadingEndpointState() {
        for (WorkerEndpoint endpoint : List.of(mock(PrefillEndpoint.class), mock(DecodeEndpoint.class))) {
            assertFalse(RouteAdmission.mustWaitForCapacity(mock(BalanceContext.class), endpoint));

            verify(endpoint).tryPinGeneration();
            verifyNoMoreInteractions(endpoint);
        }
    }

    @Test
    void decodeUsesEachRequestsOwnRequestLimit() {
        DecodeFixture fixture = decode(new CapacityUsage(1L, 1000L, 1000L, 0L, 0L));
        BalanceContext limitOne = context(1L, 1L, 90L, 5L, 5L);
        BalanceContext limitTwo = context(2L, 2L, 90L, 5L, 5L);

        assertTrue(RouteAdmission.mustWaitForCapacity(limitOne, fixture.endpoint()));
        assertFalse(RouteAdmission.mustWaitForCapacity(limitTwo, fixture.endpoint()));

        verifySnapshotAndPin(fixture, 2);
    }

    @Test
    void decodeUsesEachRequestsOwnKvThreshold() {
        DecodeFixture fixture = decode(new CapacityUsage(0L, 100L, 100L, 0L, 70L));
        BalanceContext threshold80 = context(1L, 2L, 80L, 5L, 10L);
        BalanceContext threshold90 = context(2L, 2L, 90L, 5L, 10L);

        assertTrue(RouteAdmission.mustWaitForCapacity(threshold80, fixture.endpoint()));
        assertFalse(RouteAdmission.mustWaitForCapacity(threshold90, fixture.endpoint()));

        verifySnapshotAndPin(fixture, 2);
    }

    @Test
    void decodeUsesCurrentPromptDemandForHardKvCapacity() {
        DecodeFixture fixture = decode(new CapacityUsage(0L, 1000L, 20L, 0L, 0L));
        BalanceContext largerPrompt = context(1L, 2L, 90L, 21L, 0L);
        BalanceContext smallerPrompt = context(2L, 2L, 90L, 20L, 0L);

        assertTrue(RouteAdmission.mustWaitForCapacity(largerPrompt, fixture.endpoint()));
        assertFalse(RouteAdmission.mustWaitForCapacity(smallerPrompt, fixture.endpoint()));

        verifySnapshotAndPin(fixture, 2);
    }

    @Test
    void decodeUsesCurrentOutputDemandForExpectedKvCapacity() {
        DecodeFixture fixture = decode(new CapacityUsage(0L, 100L, 100L, 0L, 80L));
        BalanceContext largerOutput = context(1L, 2L, 90L, 5L, 6L);
        BalanceContext smallerOutput = context(2L, 2L, 90L, 5L, 5L);

        assertTrue(RouteAdmission.mustWaitForCapacity(largerOutput, fixture.endpoint()));
        assertFalse(RouteAdmission.mustWaitForCapacity(smallerOutput, fixture.endpoint()));

        verifySnapshotAndPin(fixture, 2);
    }

    @Test
    void waitAtDispatchDoesNotTreatFullDecodeAsPlacementBlocked() {
        DecodeFixture fixture = decode(new CapacityUsage(64L, 100L, 0L, 0L, 100L));
        BalanceContext context = context(1L, 1L, 90L, 10L, 10L);
        SchedulingTestConfig.useFifoQueue(context.getConfig());

        assertFalse(RouteAdmission.mustWaitForCapacity(context, fixture.endpoint()));

        verify(fixture.endpoint(), never()).layeredAdmissionView();
        verify(fixture.pin()).close();
    }

    @ParameterizedTest
    @EnumSource(value = DecodeTaskPhase.class,
            names = {"MASTER_QUEUED_NOT_DISPATCHED", "ACCEPTED_NOT_RUNNING", "RUNNING"})
    void lowerPriorityOwnerLeavesReclamationToFullPlacement(DecodeTaskPhase phase) {
        DecodeFixture fixture = decode(new CapacityUsage(1L, 100L, 0L, 0L, 100L), task(phase, 10));

        assertFalse(RouteAdmission.mustWaitForCapacity(
                context(1L, 1L, 90L, 10L, 10L), fixture.endpoint()));

        verifySnapshotAndPin(fixture, 1);
    }

    @ParameterizedTest
    @EnumSource(value = DecodeTaskPhase.class,
            names = {"MASTER_QUEUED_NOT_DISPATCHED", "ACCEPTED_NOT_RUNNING", "RUNNING"})
    void unknownEqualOrHigherPriorityOwnerDoesNotRelieveCapacity(DecodeTaskPhase phase) {
        for (int priority : new int[]{0, 50, 80}) {
            DecodeFixture fixture = decode(new CapacityUsage(1L, 100L, 0L, 0L, 100L), task(phase, priority));

            assertTrue(RouteAdmission.mustWaitForCapacity(
                    context(1L, 1L, 90L, 10L, 10L), fixture.endpoint()), "owner priority " + priority);

            verifySnapshotAndPin(fixture, 1);
        }
    }

    @Test
    void snapshotFailureStillClosesTheGenerationPin() {
        DecodeFixture fixture = decode(new CapacityUsage(0L, 100L, 100L, 0L, 0L));
        IllegalStateException failure = new IllegalStateException("snapshot unavailable");
        when(fixture.endpoint().layeredAdmissionView()).thenThrow(failure);

        assertSame(failure, assertThrows(IllegalStateException.class,
                () -> RouteAdmission.mustWaitForCapacity(
                        context(1L, 1L, 90L, 5L, 5L), fixture.endpoint())));

        verifySnapshotAndPin(fixture, 1);
    }

    private static BalanceContext context(long requestId, long maxRequests, long maxKvUsagePercent,
                                          long promptTokens, long outputTokens) {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        var availability = config.getRouter().getRoles().getDecode().getAvailability();
        availability.setMaxEngineRequests(maxRequests);
        availability.setMaxKvUsagePercent(maxKvUsagePercent);
        BalanceContext context = RequestLifecycleTestSupport.context(config, requestId);
        context.getRequest().setSeqLen(promptTokens);
        context.getRequest().setMaxNewTokens(Math.toIntExact(outputTokens));
        return context;
    }

    private static DecodeRequestView task(DecodeTaskPhase phase, int priority) {
        DecodeRequestView task = mock(DecodeRequestView.class);
        when(task.phase()).thenReturn(phase);
        when(task.priority()).thenReturn(priority);
        return task;
    }

    private static DecodeFixture decode(CapacityUsage usage, DecodeRequestView... tasks) {
        DecodeEndpoint endpoint = mock(DecodeEndpoint.class);
        WorkerEndpoint.GenerationPin pin = mock(WorkerEndpoint.GenerationPin.class);
        when(endpoint.tryPinGeneration()).thenReturn(pin);
        LayeredAdmissionView view = mock(LayeredAdmissionView.class);
        DecodeRoutingView routing = mock(DecodeRoutingView.class);
        when(endpoint.layeredAdmissionView()).thenReturn(view);
        when(view.routing()).thenReturn(routing);
        when(routing.placementUsage()).thenReturn(usage);
        Map<Long, DecodeRequestView> reserved = new java.util.HashMap<>();
        List<DecodeRequestView> confirmed = new java.util.ArrayList<>();
        for (int index = 0; index < tasks.length; index++) {
            DecodeRequestView task = tasks[index];
            if (task.phase().isEngineConfirmed()) {
                confirmed.add(task);
            } else {
                reserved.put((long) index, task);
            }
        }
        when(view.reserved()).thenReturn(reserved);
        when(view.confirmed()).thenReturn(confirmed);
        return new DecodeFixture(endpoint, pin);
    }

    private static void verifySnapshotAndPin(DecodeFixture fixture, int attempts) {
        verify(fixture.endpoint(), times(attempts)).layeredAdmissionView();
        verify(fixture.pin(), times(attempts)).close();
    }

    private record DecodeFixture(DecodeEndpoint endpoint, WorkerEndpoint.GenerationPin pin) {
    }
}
