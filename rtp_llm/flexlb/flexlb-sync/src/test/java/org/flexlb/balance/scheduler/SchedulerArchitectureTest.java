package org.flexlb.balance.scheduler;

import org.flexlb.balance.eviction.DecodePreemption;
import org.flexlb.balance.planner.GroupPlanner;
import org.junit.jupiter.api.Test;
import org.springframework.asm.ClassReader;
import org.springframework.asm.ClassVisitor;
import org.springframework.asm.Handle;
import org.springframework.asm.MethodVisitor;
import org.springframework.asm.Opcodes;
import org.springframework.asm.Type;

import java.io.IOException;
import java.lang.reflect.Modifier;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Protect the migrated consumer boundaries, without constraining unrelated endpoint APIs. */
class SchedulerArchitectureTest {
    @Test
    void selectionConsumersDependOnContractsInsteadOfStrategyImplementations() throws IOException {
        assertNoImplementationDependency(
                RequestWorkerSelector.class,
                org.flexlb.balance.strategy.CostBasedPrefillStrategy.class);
        assertNoImplementationDependency(
                RequestWorkerSelector.class, org.flexlb.balance.strategy.DecodeSelector.class);
        assertNoImplementationDependency(
                org.flexlb.balance.endpoint.EndpointRegistry.PrefillRoutingEntry.class,
                org.flexlb.balance.strategy.CostBasedPrefillStrategy.class);
    }

    @Test
    void deliveryExecutionDoesNotRegisterMonitoringBackends() throws IOException {
        for (Class<?> owner :
                new Class<?>[] {DeliveryExecution.class, SchedulerTestServices.class}) {
            for (Class<?> backend :
                    new Class<?>[] {
                        io.micrometer.core.instrument.MeterRegistry.class,
                        io.micrometer.core.instrument.Gauge.class,
                        io.micrometer.core.instrument.FunctionCounter.class
                    }) {
                assertNoImplementationDependency(owner, backend);
            }
        }
    }

    @Test
    void queueExecutionDoesNotDefineOrderingOrCollectionPolicy() throws IOException {
        assertNoImplementationDependency(
                RequestExecution.class, org.flexlb.util.PriorityOrdering.class);
        inspectCalls(
                RequestExecution.class,
                (owner, name) ->
                        assertFalse(
                                owner.equals(Type.getInternalName(GroupPlanner.class))
                                        && name.equals("collectionDeadlineMs"),
                                "the grouping policy owns its next wakeup time"));
    }

    @Test
    void routingStrategiesCannotAcquirePinsOrReportBusinessOutcomes() throws IOException {
        for (Class<?> strategy :
                new Class<?>[] {
                    org.flexlb.balance.strategy.CostBasedPrefillStrategy.class,
                    org.flexlb.balance.strategy.DecodeSelector.class
                }) {
            for (Class<?> dependency :
                    new Class<?>[] {
                        org.flexlb.balance.endpoint.EndpointRegistry.class,
                        org.flexlb.balance.endpoint.PrefillEndpoint.class,
                        org.flexlb.balance.endpoint.WorkerEndpoint.GenerationPin.class,
                        RequestRoute.class,
                        org.flexlb.service.monitor.EngineHealthReporter.class,
                        org.flexlb.cache.monitor.CacheMetricsReporter.class
                    }) {
                assertNoImplementationDependency(strategy, dependency);
            }
        }
    }

    @Test
    void queueExecutionCannotChooseFacilitiesFromAnIndividualRequest() throws IOException {
        assertNoImplementationDependency(RequestExecution.class, SchedulerTestServices.class);
        inspectCalls(
                RequestExecution.class,
                (owner, name) ->
                        assertFalse(
                                owner.equals(Type.getInternalName(RequestLifecycle.class))
                                        && name.equals("runtime"),
                                "queue facilities must be bound before publication"));
    }

    @Test
    void telemetryConsumesValuesInsteadOfRequestOwnership() throws IOException {
        for (Class<?> reporter :
                new Class<?>[] {
                    org.flexlb.service.monitor.DeliveryMetricsReporter.class,
                    org.flexlb.service.monitor.EngineHealthReporter.class
                }) {
            for (Class<?> dependency :
                    new Class<?>[] {
                        RequestContext.class,
                        RequestRoute.class,
                        org.flexlb.balance.endpoint.PrefillState.class
                    }) {
                assertNoImplementationDependency(reporter, dependency);
            }
        }
    }

    @Test
    void placementAndDeliveryConsumersCannotRetainOrAcceptTheWholeScheduler() {
        for (Class<?> consumer :
                new Class<?>[] {
                    DirectRequestScheduler.class,
                    QueuedRequestScheduler.class,
                    DeliveryDispatcher.class
                }) {
            assertNoDependency(consumer, RequestScheduler.class);
        }
        assertNoDependency(RequestLifecycle.class, RequestWorkerSelector.class);
    }

    @Test
    void defaultSchedulersShareWiringWithoutInheritingRequestLifecycle() {
        assertTrue(
                Arrays.stream(GroupPlanner.class.getDeclaredFields())
                        .allMatch(f -> Modifier.isStatic(f.getModifiers())));
        assertTrue(RequestScheduler.class.isAssignableFrom(DirectRequestScheduler.class));
        assertTrue(RequestScheduler.class.isAssignableFrom(QueuedRequestScheduler.class));
        org.junit.jupiter.api.Assertions.assertEquals(
                AbstractRequestScheduler.class, DirectRequestScheduler.class.getSuperclass());
        org.junit.jupiter.api.Assertions.assertEquals(
                AbstractRequestScheduler.class, QueuedRequestScheduler.class.getSuperclass());
        assertFalse(RequestLifecycle.class.isAssignableFrom(AbstractRequestScheduler.class));
        assertTrue(AutoCloseable.class.isAssignableFrom(DirectRequestScheduler.class));
    }

    @Test
    void schedulingContractExposesTimingHooksButKeepsSettlementInTheSharedLifecycle() {
        org.junit.jupiter.api.Assertions.assertEquals(
                java.util.Set.of(
                        "onRequest",
                        "onRequestChanged",
                        "onEndpointChanged",
                        "schedule",
                        "close"),
                Arrays.stream(RequestScheduler.class.getDeclaredMethods())
                        .map(java.lang.reflect.Method::getName)
                        .collect(java.util.stream.Collectors.toSet()));
        for (Class<?> type :
                new Class<?>[] {
                    RequestLifecycle.class,
                    DirectRequestScheduler.class,
                    QueuedRequestScheduler.class
                }) {
            assertFalse(
                    Arrays.stream(type.getInterfaces())
                            .anyMatch(i -> i.getSimpleName().equals("Control")));
        }
        assertFalse(
                Arrays.stream(RequestLifecycle.class.getDeclaredMethods())
                        .anyMatch(
                                method ->
                                        java.util.Set.of(
                                                                                        "schedule",
                                                        "commitDirectRoute",
                                                        "enqueueRoute")
                                                .contains(method.getName())));
    }

    @Test
    void defaultSchedulersDoNotExposeRequestAdmissionOrCancellation() {
        for (var type : java.util.List.of(DirectRequestScheduler.class, QueuedRequestScheduler.class)) {
            assertFalse(Arrays.stream(type.getMethods())
                    .anyMatch(method -> java.util.Set.of("submit", "cancel", "lifecycle").contains(method.getName())));
        }
    }

    @Test
    void commonProtocolDoesNotCallOrCastToTheQueueImplementation() throws IOException {
        assertNoImplementationDependency(RequestLifecycle.class, QueuedRequestScheduler.class);
        assertNoImplementationDependency(RequestContext.class, QueuedRequestScheduler.class);
    }

    @Test
    void decodePreemptionUsesSchedulerCapabilityInsteadOfQueueImplementation() throws IOException {
        assertNoImplementationDependency(DecodePreemption.class, QueuedRequestScheduler.class);
        assertNoImplementationDependency(
                org.flexlb.balance.eviction.DefaultDecodePreemption.class,
                QueuedRequestScheduler.class);
        assertNoImplementationDependency(
                QueuedRequestScheduler.class,
                org.flexlb.balance.eviction.DefaultDecodePreemption.class);
        assertNoImplementationDependency(
                org.flexlb.balance.eviction.DefaultDecodePreemption.class,
                org.flexlb.balance.eviction.PriorityDecodePreemptionPolicy.class);
    }

    @Test
    void decodePreemptionDoesNotOwnPlacementPlanCompletion() {
        assertFalse(
                Arrays.stream(DecodePreemption.class.getDeclaredClasses())
                        .anyMatch(type -> type.getSimpleName().equals("AdmissionTarget")));
        assertFalse(
                Arrays.stream(DecodePreemption.class.getDeclaredMethods())
                        .anyMatch(method -> method.getName().equals("admit")));
    }

    @Test
    void routeConstructionCannotInitializeRequestQueueOrder() throws IOException {
        inspectCalls(
                RequestRoute.class,
                (owner, name) ->
                        assertFalse(
                                owner.equals(Type.getInternalName(RequestContext.class))
                                        && name.equals("initializeExecutionOrder"),
                                "RequestRoute must not assign request FIFO identity"));
    }

    @Test
    void lifecycleCannotChooseQueueRulesOrInterpretTransportReplies() throws IOException {
        for (Class<?> dependency :
                new Class<?>[] {
                    OrderedRequestQueue.class,
                    org.flexlb.balance.endpoint.PrefillActiveIndex.class,
                    QueueExecutionSettings.class,
                    RequestExecution.Dependencies.class,
                    org.flexlb.engine.grpc.EngineGrpcClient.CancelReply.class,
                    org.flexlb.balance.eviction.DefaultDecodePreemption.class,
                    org.flexlb.balance.planner.BuiltinGroupPlanner.class
                }) {
            assertNoImplementationDependency(RequestLifecycle.class, dependency);
        }
        assertNoDependency(RequestLifecycle.class, org.flexlb.config.FlexlbConfig.class);
        inspectCalls(
                RequestLifecycle.class,
                (owner, name) ->
                        assertFalse(
                                owner.equals(
                                                Type.getInternalName(
                                                        org.flexlb.config.FlexlbConfig.class))
                                        && java.util.Set.of(
                                                        "isQueue",
                                                        "isDirect",
                                                        "queueScheduler",
                                                        "decisionPolicy")
                                                .contains(name),
                                "lifecycle must not choose scheduling or queue rules"));
        inspectCalls(
                RequestLifecycle.class,
                (owner, name) ->
                        assertFalse(
                                owner.equals(Type.getInternalName(RequestData.class))
                                        && name.equals("initializeExecutionOrder"),
                                "queue entry owns its FIFO identity"));
    }

    @Test
    void schedulingAndQueuesCannotSelectConcreteStrategiesOrMonitoringBackends()
            throws IOException {
        for (Class<?> consumer :
                new Class<?>[] {
                    DirectRequestScheduler.class, QueuedRequestScheduler.class,
                    OrderedRequestQueue.class, org.flexlb.balance.endpoint.PrefillActiveIndex.class
                }) {
            for (Class<?> dependency :
                    new Class<?>[] {
                        org.flexlb.balance.strategy.CostBasedPrefillStrategy.class,
                        org.flexlb.balance.strategy.DecodeSelector.class,
                        org.flexlb.balance.strategy.VitWorkerSelector.class,
                        io.micrometer.core.instrument.MeterRegistry.class,
                        org.flexlb.metric.MicrometerFlexMonitor.class
                    }) {
                assertNoImplementationDependency(consumer, dependency);
            }
        }
    }

    private static void assertNoImplementationDependency(Class<?> consumer, Class<?> forbidden)
            throws IOException {
        assertNoDependency(consumer, forbidden);
        String forbiddenName = Type.getInternalName(forbidden);
        inspectCalls(
                consumer,
                (owner, name) ->
                        assertFalse(
                                owner.equals(forbiddenName)
                                        || owner.startsWith(forbiddenName + "$"),
                                consumer.getSimpleName() + " references " + owner + "." + name));
    }

    /** Include method bodies and method-reference handles, which reflection cannot inspect. */
    private static void inspectCalls(
            Class<?> consumer, java.util.function.BiConsumer<String, String> check)
            throws IOException {
        try (var bytecode =
                consumer.getResourceAsStream("/" + Type.getInternalName(consumer) + ".class")) {
            org.junit.jupiter.api.Assertions.assertNotNull(bytecode);
            new ClassReader(bytecode)
                    .accept(
                            new ClassVisitor(Opcodes.ASM9) {
                                @Override
                                public MethodVisitor visitMethod(
                                        int access,
                                        String name,
                                        String descriptor,
                                        String signature,
                                        String[] exceptions) {
                                    return new MethodVisitor(Opcodes.ASM9) {
                                        @Override
                                        public void visitTypeInsn(int opcode, String type) {
                                            check.accept(type, "type instruction");
                                        }

                                        @Override
                                        public void visitFieldInsn(
                                                int opcode,
                                                String owner,
                                                String name,
                                                String descriptor) {
                                            check.accept(owner, name);
                                        }

                                        @Override
                                        public void visitMethodInsn(
                                                int opcode,
                                                String owner,
                                                String name,
                                                String descriptor,
                                                boolean isInterface) {
                                            check.accept(owner, name);
                                        }

                                        @Override
                                        public void visitInvokeDynamicInsn(
                                                String name,
                                                String descriptor,
                                                Handle bootstrap,
                                                Object... arguments) {
                                            check.accept(bootstrap.getOwner(), bootstrap.getName());
                                            for (Object argument : arguments) {
                                                if (argument instanceof Handle handle) {
                                                    check.accept(
                                                            handle.getOwner(), handle.getName());
                                                }
                                            }
                                        }
                                    };
                                }
                            },
                            ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES);
        }
        for (Class<?> nested : consumer.getDeclaredClasses()) {
            inspectCalls(nested, check);
        }
    }

    private static void assertNoDependency(Class<?> consumer, Class<?> forbidden) {
        for (var field : consumer.getDeclaredFields()) {
            assertFalse(field.getType() == forbidden, field.toString());
        }
        for (var constructor : consumer.getDeclaredConstructors()) {
            assertFalse(
                    Arrays.asList(constructor.getParameterTypes()).contains(forbidden),
                    constructor.toString());
        }
        for (var method : consumer.getDeclaredMethods()) {
            assertFalse(method.getReturnType() == forbidden, method.toString());
            assertFalse(
                    Arrays.asList(method.getParameterTypes()).contains(forbidden),
                    method.toString());
        }
        for (Class<?> nested : consumer.getDeclaredClasses()) {
            assertNoDependency(nested, forbidden);
        }
    }
}
