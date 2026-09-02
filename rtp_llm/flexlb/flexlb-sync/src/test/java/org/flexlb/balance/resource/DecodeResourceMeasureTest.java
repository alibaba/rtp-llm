package org.flexlb.balance.resource;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class DecodeResourceMeasureTest {

    @Mock
    private ConfigService configService;

    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        when(configService.loadBalanceConfig()).thenReturn(config);
    }

    @Test
    void concurrency_limit_disabled_should_not_affect_decode_availability() {
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests(null);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        DecodeEndpoint endpoint = createAliveDecodeEndpoint();
        ResourceTestSupport.publish(
                endpoint.getStatus(), true, 100L, 100L,
                taskMap(1L, 2L, 3L, 4L));

        assertTrue(measure.isResourceAvailable(endpoint.routingView()));
    }

    @Test
    void worker_should_be_unavailable_when_decode_concurrency_limit_reached() {
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests(2L);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        DecodeEndpoint endpoint = createAliveDecodeEndpoint();
        reserve(endpoint, 1L, 0, 0);
        reserve(endpoint, 2L, 0, 0);
        // engineLoad = confirmedEngineOwned(0) + inflight(2) = 2;
        // limit = 2, so 2 >= 2 is unavailable.
        assertFalse(measure.isResourceAvailable(endpoint.routingView()));
    }

    @Test
    void worker_should_be_available_when_inflight_below_concurrency_limit() {
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests(3L);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        DecodeEndpoint endpoint = createAliveDecodeEndpoint();
        reserve(endpoint, 1L, 0, 0);
        // engineLoad = confirmedEngineOwned(0) + inflight(1) = 1;
        // limit = 3, so 1 < 3 is available.
        assertTrue(measure.isResourceAvailable(endpoint.routingView()));
    }

    @Test
    void routingViewExcludesQueuedReservationsFromEngineFacingLoad() {
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests(1L);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        DecodeEndpoint endpoint = createAliveDecodeEndpoint();
        reserveQueued(endpoint, 1L, 0, 0, 50);

        assertEquals(1, endpoint.routingView().totalLoad());
        assertEquals(0, endpoint.routingView().engineLoad());
        assertTrue(measure.isResourceAvailable(endpoint.routingView()));

        reserve(endpoint, 2L, 0, 0);
        assertEquals(1, endpoint.routingView().engineLoad());
        assertFalse(measure.isResourceAvailable(endpoint.routingView()));
    }

    @Test
    void dispatchPreferenceExcludesQueuedKvButChargesTheActivePermit() {
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests(2L);
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxKvUsagePercent(50L);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        DecodeEndpoint endpoint = createAliveDecodeEndpoint();
        reserveQueued(endpoint, 1L, 60, 60, 50);

        assertFalse(measure.isResourceAvailable(endpoint.routingView()),
                "hard placement accounting retains queued expected KV");
        assertTrue(measure.isEngineDispatchAvailable(endpoint.routingView()),
                "a queued shadow alone must not hide an idle Decode");

        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition =
                endpoint.acquireEngineDispatchPermit(1L, 2L, 100L);
        assertEquals(
                DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED,
                acquisition.status());
        assertFalse(measure.isEngineDispatchAvailable(endpoint.routingView()),
                "the preference must charge the exact permit's Engine-facing KV");
        assertTrue(acquisition.permit().release());
    }

    private DecodeEndpoint createAliveDecodeEndpoint() {
        WorkerStatus status = createAliveWorkerStatus();
        return new DecodeEndpoint(status, mock(EndpointEventSink.class));
    }

    private WorkerStatus createAliveWorkerStatus() {
        return ResourceTestSupport.worker(
                org.flexlb.dao.route.RoleType.DECODE,
                100L, 100L, Map.of());
    }

    private static void reserve(
            DecodeEndpoint endpoint,
            long requestId,
            long kvTokens,
            long expectedKvTokens) {
        try (var pin = endpoint.tryPinGeneration()) {
            endpoint.reservePinned(
                    pin, requestId, kvTokens, expectedKvTokens, 0);
        }
    }

    private static void reserveQueued(
            DecodeEndpoint endpoint,
            long requestId,
            long kvTokens,
            long expectedKvTokens,
            int priority) {
        try (var pin = endpoint.tryPinGeneration()) {
            endpoint.reserveQueuedPinned(
                    pin, requestId, kvTokens, expectedKvTokens, priority);
        }
    }

    private Map<String, TaskInfo> taskMap(Long... requestIds) {
        return Arrays.stream(requestIds)
                .collect(Collectors.toMap(String::valueOf, this::taskInfo));
    }

    private TaskInfo taskInfo(long requestId) {
        TaskInfo taskInfo = new TaskInfo();
        taskInfo.setRequestId(requestId);
        return taskInfo;
    }
}
