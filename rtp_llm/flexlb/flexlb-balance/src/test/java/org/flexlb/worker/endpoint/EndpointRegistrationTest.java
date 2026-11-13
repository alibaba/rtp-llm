package org.flexlb.worker.endpoint;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Verifies that all endpoints report registration outcomes with the same payload rules.
 */
class EndpointRegistrationTest {
    @Test
    void onlySuccessfulRegistrationCarriesARequestRecord() {
        var record = new Object();
        var registered =
                new WorkerEndpoint.Registration<>(
                        WorkerEndpoint.RegistrationStatus.REGISTERED, record);
        assertSame(record, registered.value());
        for (var status : WorkerEndpoint.RegistrationStatus.values()) {
            if (status == WorkerEndpoint.RegistrationStatus.REGISTERED) {
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new WorkerEndpoint.Registration<>(status, null));
            } else {
                assertEquals(status, WorkerEndpoint.Registration.rejected(status).status());
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new WorkerEndpoint.Registration<>(status, record));
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(
            value = org.flexlb.config.service.RoleType.class,
            names = {"PREFILL", "DECODE", "ENCODER"})
    void eachEndpointRollsBackOnlyItsExactUncommittedRegistration(
            org.flexlb.config.service.RoleType role) {
        var status = EndpointTestSupport.workerStatus(role, "127.0.0.1", 8001, 8002);
        var repository =
                org.flexlb.balance.schedule.SchedulerTestSupport.repository(
                        EndpointTestSupport.noopEventSink());
        WorkerEndpoint endpoint;
        java.util.function.Function<String, WorkerEndpoint.RequestRegistration> register;
        switch (role) {
            case PREFILL -> {
                var prefill =
                        new PrefillEndpoint(
                                status,
                                new org.flexlb.config.FlexlbConfig(),
                                org.mockito.Mockito.mock(
                                        org.flexlb.worker.endpoint.CapacityNotifier.class));
                endpoint = prefill;
                register =
                        id ->
                                prefill.tryRegisterSelection(id, prefill.selectionSnapshot(), 10)
                                        .value();
            }
            case DECODE -> {
                var decode = EndpointTestSupport.decode(status, repository);
                endpoint = decode;
                register =
                        id ->
                                decode.tryRegisterSelection(
                                                decode.selectionSnapshot(),
                                                id,
                                                10,
                                                10,
                                                50,
                                                new DecodeResources.AdmissionCapacity(4, 100))
                                        .value();
            }
            case ENCODER -> {
                var encoder =
                        new EncoderEndpoint(
                                status,
                                org.mockito.Mockito.mock(
                                        org.flexlb.worker.endpoint.CapacityNotifier.class));
                endpoint = encoder;
                register = id -> encoder.tryRegisterSelection(id, 4, 10).value();
            }
            default -> throw new AssertionError(role);
        }
        try {
            SelectionRollback rollback =
                    org.junit.jupiter.api.Assertions.assertInstanceOf(
                            SelectionRollback.class, endpoint);
            var original = register.apply("request");
            org.junit.jupiter.api.Assertions.assertNotNull(original);
            org.junit.jupiter.api.Assertions.assertTrue(rollback.rollbackSelection(original));
            org.junit.jupiter.api.Assertions.assertFalse(rollback.rollbackSelection(original));
            var replacement = register.apply("request");
            org.junit.jupiter.api.Assertions.assertNotNull(replacement);
            org.junit.jupiter.api.Assertions.assertFalse(rollback.rollbackSelection(original));
            org.junit.jupiter.api.Assertions.assertTrue(rollback.rollbackSelection(replacement));
        } finally {
            EndpointTestSupport.closeAndWait(endpoint);
        }
    }
}
