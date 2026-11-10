package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;
import org.springframework.asm.ClassReader;
import org.springframework.asm.ClassVisitor;
import org.springframework.asm.MethodVisitor;
import org.springframework.asm.Opcodes;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.CALLS_REAL_METHODS;
import static org.mockito.Mockito.mock;

/**
 * Keeps delivery completion separate from prediction, preparation and remote cleanup.
 */
class DeliveryReportingBoundaryTest {
    @Test
    void dispatcherReportsAsyncResultsAndSchedulersConsumeReturnedRouteResults() throws Exception {
        Path classes =
                Path.of(
                        DeliveryCoordinator.class
                                .getProtectionDomain()
                                .getCodeSource()
                                .getLocation()
                                .toURI());
        var calls = new AtomicInteger();
        try (var files = Files.walk(classes.resolve("org/flexlb/balance"))) {
            for (var file : files.filter(path -> path.toString().endsWith(".class")).toList()) {
                inspect(file, calls);
            }
        }
        assertEquals(
                3, calls.get(), "one dispatcher report and two synchronous route-result consumers");
    }

    @Test
    void predictionPreparationAndCleanupAreNotDeliveryResults() {
        var scheduler = mock(AbstractRequestScheduler.class, CALLS_REAL_METHODS);
        assertThrows(
                IllegalArgumentException.class,
                () -> scheduler.onDeliveryResult(new ContextResult.Prediction(null, 100L, 1L)));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        scheduler.onDeliveryResult(
                                new ContextResult.PreparationFailure(null, null, "failed", 1L)));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        scheduler.onDeliveryResult(
                                new ContextResult.CleanupAcknowledgement(null, null, 1L)));
    }

    private static void inspect(Path file, AtomicInteger calls) throws IOException {
        new ClassReader(Files.readAllBytes(file))
                .accept(
                        new ClassVisitor(Opcodes.ASM9) {
                            private String className;

                            @Override
                            public void visit(
                                    int version,
                                    int access,
                                    String name,
                                    String signature,
                                    String superName,
                                    String[] interfaces) {
                                className = name;
                            }

                            @Override
                            public MethodVisitor visitMethod(
                                    int access,
                                    String name,
                                    String descriptor,
                                    String signature,
                                    String[] exceptions) {
                                return new MethodVisitor(Opcodes.ASM9) {
                                    @Override
                                    public void visitMethodInsn(
                                            int opcode,
                                            String owner,
                                            String called,
                                            String descriptor,
                                            boolean isInterface) {
                                        if (className.equals(
                                                "org/flexlb/balance/scheduler/DeliveryDispatcher"))
                                            assertNotEquals(
                                                    "org/flexlb/balance/scheduler/DeliveryCoordinator",
                                                    owner,
                                                    "Dispatcher must complete delivery without calling back into Coordinator");
                                        if (!called.equals("onDeliveryResult")) return;
                                        calls.incrementAndGet();
                                        var permittedCallers =
                                                java.util.Map.of(
                                                        "org/flexlb/balance/scheduler/DeliveryDispatcher",
                                                                "report",
                                                        "org/flexlb/balance/scheduler/DirectRequestScheduler",
                                                                "selectAndPublish",
                                                        "org/flexlb/balance/scheduler/QueuedRequestScheduler",
                                                                "executeQueuedRouting");
                                        assertEquals(
                                                permittedCallers.get(className),
                                                name,
                                                "Only Dispatcher reports asynchronous delivery; schedulers consume returned route results");
                                    }
                                };
                            }
                        },
                        ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES);
    }
}
