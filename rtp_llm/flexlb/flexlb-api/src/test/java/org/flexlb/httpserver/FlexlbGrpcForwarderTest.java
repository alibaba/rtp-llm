package org.flexlb.httpserver;

import io.grpc.CallOptions;
import io.grpc.ManagedChannel;
import io.grpc.MethodDescriptor;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.netty.channel.EventLoopGroup;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.lang.reflect.Field;
import java.util.Map;
import java.util.concurrent.Executor;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class FlexlbGrpcForwarderTest {

    @Test
    void missingMasterIsTheOnlyLocalFallbackCase() {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);

        FlexlbGrpcForwarder.MasterForwardResult result =
                forwarder.forwardToMaster(request(1L));

        assertFalse(result.masterFound());
        assertNull(result.response());
        assertEquals("MASTER_NULL", result.failure());
        verify(reporter).reportForwardToMasterResult("LOCAL", "MASTER_NULL");
    }

    @Test
    void grpcFailureDoesNotSetIndependentDeadlineAndKeepsChannelUntilShutdown()
            throws Exception {
        LBStatusConsistencyService consistency = masterAt("10.0.0.2:7001");
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        FlexlbGrpcForwarder forwarder = forwarder(consistency, reporter);
        ManagedChannel channel = mock(ManagedChannel.class);
        ArgumentCaptor<CallOptions> callOptions = ArgumentCaptor.forClass(CallOptions.class);
        when(channel.newCall(any(MethodDescriptor.class), callOptions.capture()))
                .thenThrow(new StatusRuntimeException(Status.DEADLINE_EXCEEDED));
        channels(forwarder).put("10.0.0.2:7003", channel);

        FlexlbGrpcForwarder.MasterForwardResult result =
                forwarder.forwardToMaster(request(4L));

        assertTrue(result.masterFound());
        assertEquals("DEADLINE_EXCEEDED", result.failure());
        assertNull(callOptions.getValue().getDeadline());
        assertSame(channel, channels(forwarder).get("10.0.0.2:7003"));
        verify(channel, never()).shutdownNow();
        verify(reporter).reportForwardToMasterResult("10.0.0.2", "GRPC_FAILED");

        forwarder.shutdown();
        verify(channel).shutdownNow();
        assertTrue(channels(forwarder).isEmpty());
    }

    private static FlexlbGrpcForwarder forwarder(
            LBStatusConsistencyService consistency,
            EngineHealthReporter reporter) {
        return new FlexlbGrpcForwarder(consistency, mock(ConfigService.class), reporter,
                mock(EventLoopGroup.class), mock(Executor.class));
    }

    private static LBStatusConsistencyService masterAt(String address) {
        LBStatusConsistencyService consistency = mock(LBStatusConsistencyService.class);
        when(consistency.getMasterHostIpPort()).thenReturn(address);
        return consistency;
    }

    private static FlexlbScheduleProtocol.FlexlbScheduleRequestPB request(long id) {
        return FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(id)
                .build();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, ManagedChannel> channels(
            FlexlbGrpcForwarder forwarder) throws Exception {
        Field field = FlexlbGrpcForwarder.class.getDeclaredField("channels");
        field.setAccessible(true);
        return (Map<String, ManagedChannel>) field.get(forwarder);
    }
}
