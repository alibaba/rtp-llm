package org.flexlb.balance.scheduler;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.preemption.DecodePreemption;
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
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Protect the migrated consumer boundaries, without constraining unrelated endpoint APIs.
 */
class SchedulerArchitectureTest {
    @Test
    void dispatcherOwnsDeliveryCommitWithoutCallingBackIntoScheduler() throws IOException {
        var permitType =
                Type.getInternalName(
                        org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermit.class);
        for (var scheduler :
                java.util.List.of(
                        AbstractRequestScheduler.class,
                        DirectRequestScheduler.class,
                        QueuedRequestScheduler.class)) {
            for (var method : scheduler.getDeclaredMethods()) {
                assertFalse(
                        method.getName().equals("commitDelivery"),
                        "Scheduler must not expose a delivery execution entry point: " + method);
            }
            inspectCalls(
                    scheduler,
                    (owner, name) -> {
                        assertFalse(
                                owner.equals(permitType) && name.equals("dispatch"),
                                "Dispatcher transfers delivery permits under the request lock");
                        assertFalse(
                                owner.equals(Type.getInternalName(RequestContext.class))
                                        && java.util.Set.of(
                                                        "validateDelivery",
                                                        "recordDeliveryStarted",
                                                        "recordDirectDeliveryStarted")
                                                .contains(name),
                                "Scheduler must submit delivery through Dispatcher: " + name);
                    });
        }
        var dispatcherCommits = new AtomicBoolean();
        assertNoImplementationDependency(DeliveryDispatcher.class, DeliveryCoordinator.class);
        inspectCalls(
                DeliveryDispatcher.class,
                (owner, name) -> {
                    if (owner.equals(permitType) && name.equals("dispatch"))
                        dispatcherCommits.set(true);
                    if (owner.equals(Type.getInternalName(AbstractRequestScheduler.class))) {
                        assertFalse(
                                java.util.Set.of(
                                                "commitDelivery",
                                                "onRouteFailure",
                                                "cancel",
                                                "closeRequest",
                                                "recordDeadline",
                                                "isCurrentRequest")
                                        .contains(name),
                                "Dispatcher reports delivery outcomes instead of delegating commit or lifecycle policy: "
                                        + name);
                    }
                });
        assertTrue(dispatcherCommits.get(), "Dispatcher must perform the resource transfer");
    }

    @Test
    void prefillEndpointDoesNotReadOrModifyDispatcherPreparation() throws IOException {
        assertNoImplementationDependency(
                org.flexlb.balance.endpoint.PrefillEndpoint.class, Delivery.class);
    }

    @Test
    void queueSchedulingDoesNotConstructDeliveryPreparation() throws IOException {
        inspectCalls(
                PrefillQueueScheduler.class,
                (owner, name) ->
                        assertFalse(
                                owner.equals(Type.getInternalName(Delivery.class))
                                        && name.equals("<init>"),
                                "DeliveryCoordinator owns construction and cleanup of delivery preparation"));
        for (var method : DeliveryCoordinator.class.getMethods()) {
            if (method.getName().equals("submit"))
                assertFalse(
                        java.util.Arrays.asList(method.getParameterTypes())
                                .contains(Delivery.class));
        }
    }

    @Test
    void defaultSchedulingDoesNotOwnDeliveryFacilities() throws IOException {
        assertNoImplementationDependency(QueuedRequestScheduler.class, DeliveryExecution.class);
        assertNoImplementationDependency(PrefillQueueScheduler.class, DeliveryExecution.class);
        assertNoImplementationDependency(Delivery.class, DeliveryExecution.class);
    }

    @Test
    void sharedLifecycleDoesNotImplementSchedulingPolicies() throws IOException {
        assertNoImplementationDependency(
                AbstractRequestScheduler.class, QueueExecutionSettings.class);
        assertNoImplementationDependency(AbstractRequestScheduler.class, DecodePreemption.class);
        var policyMethods =
                java.util.Set.of(
                        "selectAndPublish",
                        "executeRouting",
                        "planRoute",
                        "resumeRoute",
                        "enqueueRoute",
                        "notifyQueuedRoute",
                        "commitDirectRoute",
                        "withdrawFailedRoute");
        for (var method : AbstractRequestScheduler.class.getDeclaredMethods()) {
            assertFalse(policyMethods.contains(method.getName()), method.toString());
        }
        inspectCalls(
                AbstractRequestScheduler.class,
                (owner, name) -> {
                    if (owner.equals(
                            Type.getInternalName(
                                    org.flexlb.balance.endpoint.PrefillEndpoint.class))) {
                        assertFalse(
                                java.util.Set.of(
                                                "commitDirect",
                                                "enqueueRoute",
                                                "signal",
                                                "removeQueued")
                                        .contains(name),
                                "the concrete scheduler owns route publication and queue retry");
                    }
                });
    }

    @Test
    void responseExecutorDoesNotChooseResponsesOrReportDeliveryOutcomes() throws IOException {
        for (Class<?> facility :
                new Class<?>[] {
                    ResponseCompletionExecutor.class, ResponseCompletionExecutor.Registration.class
                }) {
            assertNoImplementationDependency(
                    facility, org.flexlb.service.monitor.DeliveryMetricsReporter.class);
            inspectCalls(
                    facility,
                    (owner, name) ->
                            assertFalse(
                                    owner.equals(Type.getInternalName(RequestContext.class))
                                            && name.equals("selectResponse"),
                                    "AbstractRequestScheduler owns response arbitration; the executor only runs accepted work"));
        }
    }

    @Test
    void endpointDoesNotConstructOrDriveTheDefaultScheduler() throws IOException {
        assertNoImplementationDependency(
                org.flexlb.balance.endpoint.PrefillEndpoint.class, QueuedRequestScheduler.class);
        assertNoImplementationDependency(
                org.flexlb.balance.endpoint.PrefillEndpoint.class, PrefillQueueScheduler.class);
        assertNoImplementationDependency(
                org.flexlb.balance.endpoint.PrefillEndpoint.class, DeliveryCoordinator.class);
    }

    @Test
    void endpointQueueClosureDoesNotSettleRequests() throws IOException {
        inspectCalls(
                org.flexlb.balance.endpoint.PrefillEndpoint.class,
                (owner, name) ->
                        assertFalse(
                                owner.equals(Type.getInternalName(AbstractRequestScheduler.class))
                                        && name.equals("rejectRoute"),
                                "Scheduler settles drained requests; Endpoint only returns their exact membership"));
    }

    @Test
    void directSchedulingDoesNotDependOnQueueOrPreemptionPolicy() throws IOException {
        assertNoImplementationDependency(
                DirectRequestScheduler.class, QueueExecutionSettings.class);
        assertNoImplementationDependency(
                DirectRequestScheduler.class, QueuedRequestScheduler.class);
        assertNoImplementationDependency(DirectRequestScheduler.class, DecodePreemption.class);
    }

    @Test
    void queueImplementationsDoNotOwnResourcesOrRequestSettlement() throws IOException {
        for (Class<?> queue :
                new Class<?>[] {
                    OrderedRequestQueue.class, org.flexlb.balance.endpoint.PrefillRequestQueue.class
                }) {
            for (Class<?> owner :
                    new Class<?>[] {
                        AbstractRequestScheduler.class,
                        DeliveryCoordinator.class,
                        org.flexlb.balance.endpoint.PrefillEndpoint.class,
                        org.flexlb.balance.endpoint.DecodeEndpoint.class
                    }) {
                assertNoImplementationDependency(queue, owner);
            }
        }
    }

    @Test
    void selectionConsumersDependOnContractsInsteadOfStrategyImplementations() throws IOException {
        assertNoImplementationDependency(
                RequestRouteSelector.class,
                org.flexlb.balance.strategy.CostBasedPrefillStrategy.class);
        assertNoImplementationDependency(
                RequestRouteSelector.class, org.flexlb.balance.strategy.DecodeSelector.class);
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
        inspectCalls(
                PrefillQueueScheduler.class,
                (owner, name) ->
                        assertFalse(
                                owner.equals(Type.getInternalName(GroupPlanner.class))
                                        && name.equals("collectionDeadlineMs"),
                                "the grouping policy owns its next wakeup time"));
    }

    @Test
    void schedulerDoesNotOwnExternalTimerPools() throws IOException {
        assertNoImplementationDependency(AbstractRequestScheduler.class, TaskTimer.class);
        assertNoImplementationDependency(AbstractRequestScheduler.class, ScheduledTaskTimer.class);
    }

    @Test
    void lifecycleAndSchedulerDoNotCarryExecutionCapacity() {
        for (var type : new Class<?>[] {AbstractRequestScheduler.class}) {
            for (var field : type.getDeclaredFields()) {
                assertFalse(
                        org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity.class
                                .isAssignableFrom(field.getType()),
                        "execution owns its admission capacity");
            }
        }
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
                        org.flexlb.balance.endpoint.WorkerEndpoint.class,
                        RouteResult.class,
                        org.flexlb.service.monitor.EngineHealthReporter.class,
                        org.flexlb.cache.telemetry.CacheMetricsReporter.class
                    }) {
                assertNoImplementationDependency(strategy, dependency);
            }
        }
    }

    @Test
    void queueExecutionCannotChooseFacilitiesFromAnIndividualRequest() throws IOException {
        assertNoImplementationDependency(PrefillQueueScheduler.class, SchedulerTestServices.class);
        inspectCalls(
                PrefillQueueScheduler.class,
                (owner, name) ->
                        assertFalse(
                                owner.equals(Type.getInternalName(AbstractRequestScheduler.class))
                                        && name.equals("runtime"),
                                "queue facilities must be bound before publication"));
    }

    @Test
    void telemetryConsumesValuesInsteadOfRequestOwnership() throws IOException {
        for (Class<?> reporter :
                new Class<?>[] {
                    org.flexlb.service.monitor.DeliveryMetricsReporter.class,
                    org.flexlb.service.monitor.EngineHealthReporter.class,
                    org.flexlb.service.monitor.RequestSchedulerReporter.class
                }) {
            for (Class<?> dependency :
                    new Class<?>[] {
                        RequestContext.class,
                        RouteResult.class,
                        org.flexlb.balance.endpoint.PrefillEndpoint.class
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
                    DeliveryCoordinator.class
                }) {
            assertNoDependency(consumer, RequestScheduler.class);
        }
        assertNoDependency(AbstractRequestScheduler.class, RequestRouteSelector.class);
    }

    @Test
    void defaultSchedulersInheritSharedRequestManagement() {
        assertTrue(
                Arrays.stream(GroupPlanner.class.getDeclaredFields())
                        .allMatch(f -> Modifier.isStatic(f.getModifiers())));
        assertTrue(RequestScheduler.class.isAssignableFrom(DirectRequestScheduler.class));
        assertTrue(RequestScheduler.class.isAssignableFrom(QueuedRequestScheduler.class));
        org.junit.jupiter.api.Assertions.assertEquals(
                AbstractRequestScheduler.class, DirectRequestScheduler.class.getSuperclass());
        org.junit.jupiter.api.Assertions.assertEquals(
                AbstractRequestScheduler.class, QueuedRequestScheduler.class.getSuperclass());
        for (var field : AbstractRequestScheduler.class.getDeclaredFields()) {
            assertFalse(
                    RequestScheduler.class.isAssignableFrom(field.getType()),
                    "the base must not hold a second scheduler or require binding");
        }
        assertTrue(AutoCloseable.class.isAssignableFrom(DirectRequestScheduler.class));
    }

    @Test
    void schedulingContractExposesTimingHooksAndBaseImplementsRequestCleanup() {
        org.junit.jupiter.api.Assertions.assertEquals(
                java.util.Set.of(
                        "start",
                        "scheduleRequest",
                        "onEndpointChanged",
                        "allowsRoutePreemption",
                        "shouldRetryRouteFailure",
                        "stop",
                        "awaitStopped",
                        "blockedRequestCount",
                        "queuedRequestCount",
                        "close"),
                Arrays.stream(RequestScheduler.class.getDeclaredMethods())
                        .filter(method -> !method.isSynthetic())
                        .map(java.lang.reflect.Method::getName)
                        .collect(java.util.stream.Collectors.toSet()));
        for (Class<?> type :
                new Class<?>[] {
                    AbstractRequestScheduler.class,
                    DirectRequestScheduler.class,
                    QueuedRequestScheduler.class
                }) {
            assertFalse(
                    Arrays.stream(type.getInterfaces())
                            .anyMatch(i -> i.getSimpleName().equals("Control")));
        }
        assertFalse(
                Arrays.stream(AbstractRequestScheduler.class.getDeclaredMethods())
                        .anyMatch(
                                method -> java.util.Set.of("schedule").contains(method.getName())));
    }

    @Test
    void defaultSchedulersInheritAdmissionAndCancellationWithoutForwarders() throws Exception {
        for (var type :
                java.util.List.of(DirectRequestScheduler.class, QueuedRequestScheduler.class)) {
            org.junit.jupiter.api.Assertions.assertEquals(
                    AbstractRequestScheduler.class,
                    type.getMethod("submit", RequestContext.class).getDeclaringClass());
            org.junit.jupiter.api.Assertions.assertEquals(
                    AbstractRequestScheduler.class,
                    type.getMethod("cancel", RequestContext.class, long.class, CancelReason.class)
                            .getDeclaringClass());
            assertFalse(
                    Arrays.stream(type.getDeclaredMethods())
                            .anyMatch(
                                    method ->
                                            java.util.Set.of(
                                                            "submit",
                                                            "cancel",
                                                            "lifecycle",
                                                            "bindScheduler")
                                                    .contains(method.getName())));
        }
    }

    @Test
    void commonProtocolDoesNotCallOrCastToTheQueueImplementation() throws IOException {
        assertNoImplementationDependency(
                AbstractRequestScheduler.class, QueuedRequestScheduler.class);
        assertNoImplementationDependency(RequestContext.class, QueuedRequestScheduler.class);
    }

    @Test
    void decodePreemptionUsesSchedulerCapabilityInsteadOfQueueImplementation() throws IOException {
        assertNoImplementationDependency(DecodePreemption.class, QueuedRequestScheduler.class);
        assertNoImplementationDependency(
                org.flexlb.balance.preemption.DefaultDecodePreemption.class,
                QueuedRequestScheduler.class);
        assertNoImplementationDependency(
                QueuedRequestScheduler.class,
                org.flexlb.balance.preemption.DefaultDecodePreemption.class);
        assertNoImplementationDependency(
                org.flexlb.balance.preemption.DefaultDecodePreemption.class,
                org.flexlb.balance.preemption.PriorityDecodePreemptionPolicy.class);
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
                RouteResult.class,
                (owner, name) ->
                        assertFalse(
                                owner.equals(Type.getInternalName(RequestContext.class))
                                        && name.equals("initializeExecutionOrder"),
                                "RouteResult must not assign request FIFO identity"));
    }

    @Test
    void schedulerBaseCannotChooseQueueRulesOrInterpretTransportReplies() throws IOException {
        for (Class<?> dependency :
                new Class<?>[] {
                    OrderedRequestQueue.class,
                    org.flexlb.balance.endpoint.PrefillRequestQueue.class,
                    org.flexlb.engine.grpc.EngineGrpcClient.CancelReply.class,
                    org.flexlb.balance.preemption.DefaultDecodePreemption.class,
                    org.flexlb.balance.planner.BuiltinGroupPlanner.class
                }) {
            assertNoImplementationDependency(AbstractRequestScheduler.class, dependency);
        }
        assertNoDependency(AbstractRequestScheduler.class, org.flexlb.config.FlexlbConfig.class);
        inspectCalls(
                AbstractRequestScheduler.class,
                (owner, name) ->
                        assertFalse(
                                owner.equals(
                                                Type.getInternalName(
                                                        org.flexlb.config.FlexlbConfig.class))
                                        && java.util.Set.of("queueScheduler", "decisionPolicy")
                                                .contains(name),
                                "the scheduler base may validate mode but must not choose queue rules"));
        inspectCalls(
                DeliveryCoordinator.class,
                (owner, name) ->
                        assertFalse(
                                owner.equals(Type.getInternalName(AbstractRequestScheduler.class))
                                        && name.equals("finishRouting"),
                                "the scheduler owns placement completion"));
    }

    @Test
    void schedulingAndQueuesCannotSelectConcreteStrategiesOrMonitoringBackends()
            throws IOException {
        for (Class<?> consumer :
                new Class<?>[] {
                    DirectRequestScheduler.class, QueuedRequestScheduler.class,
                    OrderedRequestQueue.class, org.flexlb.balance.endpoint.PrefillRequestQueue.class
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
}
