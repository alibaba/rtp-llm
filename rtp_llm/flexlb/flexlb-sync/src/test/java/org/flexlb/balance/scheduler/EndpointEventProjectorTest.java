package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.GrpcFailure;
import org.flexlb.service.monitor.StatusFailureReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EndpointEventProjectorTest {
    @ParameterizedTest
    @MethodSource("diagnosticFailures")
    void diagnosticFailureDoesNotStopRemainingFacts(boolean decode, Throwable diagnosticFailure) {
        var requests = mock(RequestRegistry.class);
        var reporter = mock(StatusFailureReporter.class);
        var projector = new EndpointEventProjector(requests, reporter);
        var projectionFailure = new OutOfMemoryError("local projection allocation");
        doThrow(diagnosticFailure)
                .when(reporter)
                .report(
                        any(),
                        eq("worker_status"),
                        eq(Stage.STATUS_PROJECTION_FAILED),
                        eq("LOCAL_MEMORY_EXHAUSTED"),
                        any());

        if (decode) {
            var endpoint = mock(DecodeEndpoint.class);
            var first = decodeFact();
            var second = decodeFact();
            doThrow(projectionFailure).when(requests).processDecodeStatus(endpoint, first);

            assertDoesNotThrow(() -> projector.onDecodeStatus(endpoint, List.of(first, second)));

            verify(requests).processDecodeStatus(endpoint, second);
        } else {
            var endpoint = mock(PrefillEndpoint.class);
            var first = prefillFact();
            var second = prefillFact();
            doThrow(projectionFailure)
                    .when(requests)
                    .processPrefillStatus(endpoint, RoleType.PDFUSION, first);

            assertDoesNotThrow(
                    () ->
                            projector.onPrefillStatus(
                                    endpoint, RoleType.PDFUSION, List.of(first, second)));

            verify(requests).processPrefillStatus(endpoint, RoleType.PDFUSION, second);
        }
        verify(reporter)
                .report(
                        eq(decode ? RoleType.DECODE : RoleType.PDFUSION),
                        eq("worker_status"),
                        eq(Stage.STATUS_PROJECTION_FAILED),
                        eq("LOCAL_MEMORY_EXHAUSTED"),
                        argThat(
                                failure ->
                                        failure.kind() == GrpcFailure.Kind.LOCAL_MEMORY_EXHAUSTED));
    }

    @Test
    void contractFailureKeepsProjectionReasonAndUnconfirmedAttribution() {
        var requests = mock(RequestRegistry.class);
        var reporter = mock(StatusFailureReporter.class);
        var endpoint = mock(PrefillEndpoint.class);
        var fact = prefillFact();
        doThrow(new IllegalStateException("state contract"))
                .when(requests)
                .processPrefillStatus(endpoint, RoleType.PREFILL, fact);

        new EndpointEventProjector(requests, reporter)
                .onPrefillStatus(endpoint, RoleType.PREFILL, List.of(fact));

        verify(reporter)
                .report(
                        eq(RoleType.PREFILL),
                        eq("worker_status"),
                        eq(Stage.STATUS_PROJECTION_FAILED),
                        eq("STATUS_PROJECTION_FAILED"),
                        argThat(failure -> !failure.confirmedLocalFailure()));
    }

    private static PrefillState.WorkerStatusFact prefillFact() {
        var fact = mock(PrefillState.WorkerStatusFact.class);
        when(fact.item()).thenReturn(mock(RequestExecution.class));
        return fact;
    }

    private static DecodeEndpoint.WorkerStatusFact decodeFact() {
        var fact = mock(DecodeEndpoint.WorkerStatusFact.class);
        when(fact.reservation()).thenReturn(mock(DecodeEndpoint.ReservationHandle.class));
        return fact;
    }

    static Stream<Arguments> diagnosticFailures() {
        return Stream.of(false, true)
                .flatMap(
                        decode ->
                                Stream.of(
                                        Arguments.of(
                                                decode,
                                                new IllegalStateException("metrics unavailable")),
                                        Arguments.of(
                                                decode,
                                                new OutOfMemoryError("metrics allocation"))));
    }
}
