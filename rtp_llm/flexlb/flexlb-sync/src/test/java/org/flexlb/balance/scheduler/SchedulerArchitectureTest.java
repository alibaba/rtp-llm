package org.flexlb.balance.scheduler;

import org.flexlb.balance.planner.GroupingPolicy;
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
    void placementAndDeliveryConsumersCannotRetainOrAcceptTheWholeScheduler() {
        for (Class<?> consumer : new Class<?>[] {DirectRequestScheduler.class, QueuedRequestScheduler.class,
                RouteDeliveryStrategy.class, BatchDeliveryStrategy.class}) {
            assertNoDependency(consumer, RequestScheduler.class);
        }
        assertNoDependency(AbstractRequestScheduler.class, RequestWorkerSelector.class);
    }

    @Test
    void groupingPoliciesAreStatelessAndPlacementDoesNotPretendToHaveACommonLifecycle() {
        assertTrue(Arrays.stream(GroupingPolicy.class.getDeclaredFields())
                .allMatch(f -> Modifier.isStatic(f.getModifiers())));
        assertTrue(RequestScheduler.class.isAssignableFrom(DirectRequestScheduler.class));
        assertTrue(RequestScheduler.class.isAssignableFrom(QueuedRequestScheduler.class));
        assertFalse(AutoCloseable.class.isAssignableFrom(DirectRequestScheduler.class));
    }

    @Test
    void publicContractHasOnlySchedulingAndCancellation() {
        org.junit.jupiter.api.Assertions.assertEquals(java.util.Set.of("submit", "cancel"),
                Arrays.stream(RequestScheduler.class.getDeclaredMethods()).map(java.lang.reflect.Method::getName)
                        .collect(java.util.stream.Collectors.toSet()));
        for (Class<?> type : new Class<?>[] {AbstractRequestScheduler.class, DirectRequestScheduler.class,
                QueuedRequestScheduler.class}) {
            assertFalse(Arrays.stream(type.getInterfaces()).anyMatch(i -> i.getSimpleName().equals("Control")));
        }
        assertFalse(Arrays.stream(AbstractRequestScheduler.class.getDeclaredMethods())
                .anyMatch(method -> java.util.Set.of("submit", "schedule", "commitDirectRoute", "enqueueRoute")
                        .contains(method.getName())));
    }

    @Test
    void commonProtocolDoesNotCallOrCastToTheQueueImplementation() throws IOException {
        assertNoImplementationDependency(AbstractRequestScheduler.class, QueuedRequestScheduler.class);
        assertNoImplementationDependency(RequestContext.class, QueuedRequestScheduler.class);
    }

    @Test
    void routeConstructionCannotInitializeRequestQueueOrder() throws IOException {
        inspectCalls(RequestRoute.class, (owner, name) -> assertFalse(
                owner.equals(Type.getInternalName(RequestContext.class)) && name.equals("initializeWorkerQueue"),
                "RequestRoute must not assign request FIFO identity"));
    }

    private static void assertNoImplementationDependency(Class<?> consumer, Class<?> forbidden) throws IOException {
        assertNoDependency(consumer, forbidden);
        String forbiddenName = Type.getInternalName(forbidden);
        inspectCalls(consumer, (owner, name) -> assertFalse(owner.equals(forbiddenName)
                || owner.startsWith(forbiddenName + "$"), consumer.getSimpleName() + " references " + owner + "." + name));
    }

    /** Include method bodies and method-reference handles, which reflection cannot inspect. */
    private static void inspectCalls(Class<?> consumer, java.util.function.BiConsumer<String, String> check) throws IOException {
        try (var bytecode = consumer.getResourceAsStream("/" + Type.getInternalName(consumer) + ".class")) {
            org.junit.jupiter.api.Assertions.assertNotNull(bytecode);
            new ClassReader(bytecode).accept(new ClassVisitor(Opcodes.ASM9) {
                @Override
                public MethodVisitor visitMethod(int access, String name, String descriptor, String signature, String[] exceptions) {
                    return new MethodVisitor(Opcodes.ASM9) {
                        @Override
                        public void visitTypeInsn(int opcode, String type) { check.accept(type, "type instruction"); }

                        @Override
                        public void visitFieldInsn(int opcode, String owner, String name, String descriptor) {
                            check.accept(owner, name);
                        }

                        @Override
                        public void visitMethodInsn(int opcode, String owner, String name, String descriptor, boolean isInterface) {
                            check.accept(owner, name);
                        }

                        @Override
                        public void visitInvokeDynamicInsn(String name, String descriptor, Handle bootstrap, Object... arguments) {
                            check.accept(bootstrap.getOwner(), bootstrap.getName());
                            for (Object argument : arguments) {
                                if (argument instanceof Handle handle) { check.accept(handle.getOwner(), handle.getName()); }
                            }
                        }
                    };
                }
            }, ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES);
        }
        for (Class<?> nested : consumer.getDeclaredClasses()) { inspectCalls(nested, check); }
    }

    private static void assertNoDependency(Class<?> consumer, Class<?> forbidden) {
        for (var field : consumer.getDeclaredFields()) {
            assertFalse(field.getType() == forbidden, field.toString());
        }
        for (var constructor : consumer.getDeclaredConstructors()) {
            assertFalse(Arrays.asList(constructor.getParameterTypes()).contains(forbidden), constructor.toString());
        }
        for (var method : consumer.getDeclaredMethods()) {
            assertFalse(method.getReturnType() == forbidden, method.toString());
            assertFalse(Arrays.asList(method.getParameterTypes()).contains(forbidden), method.toString());
        }
        for (Class<?> nested : consumer.getDeclaredClasses()) {
            assertNoDependency(nested, forbidden);
        }
    }
}
