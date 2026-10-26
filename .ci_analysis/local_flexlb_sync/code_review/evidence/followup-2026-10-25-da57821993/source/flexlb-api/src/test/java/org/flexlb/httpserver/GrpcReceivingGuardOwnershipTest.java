package org.flexlb.httpserver;

import io.grpc.Context;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.ServerCall;
import io.grpc.ServerStreamTracer;
import io.grpc.Status;
import io.grpc.internal.SerializingExecutor;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.PooledByteBufAllocator;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.concurrent.Executor;
import java.util.concurrent.ScheduledThreadPoolExecutor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GrpcReceivingGuardOwnershipTest {
    static class Call extends ServerCall<Object, Object> {
        public void request(int n) {}

        public void sendHeaders(Metadata h) {}

        public void sendMessage(Object o) {}

        public void close(Status s, Metadata h) {}

        public boolean isCancelled() {
            return false;
        }

        public MethodDescriptor<Object, Object> getMethodDescriptor() {
            return null;
        }
    }

    @Test
    void errorCannotReturnReservationBeforeQueuedActualByteBufCleanup() {
        queued(false);
        queued(true);
    }

    void queued(boolean startFails) {
        ArrayDeque<Runnable> work = new ArrayDeque<>();
        var guard = new GrpcReceivingGuard(1, 60000, work::add);
        var allocator = new PooledByteBufAllocator(true);
        ByteBuf retained = allocator.directBuffer(65536, 65536);
        ServerStreamTracer tracer = guard.newServerStreamTracer("test/Schedule", new Metadata());
        Context c = tracer.filterContext(Context.ROOT), before = c.attach();
        try {
            Call call = new Call();
            Executor e = guard.getExecutor(call, new Metadata());
            ServerCall.Listener<Object> listener = null;
            try {
                listener =
                        guard.interceptCall(
                                call,
                                new Metadata(),
                                (x, h) -> {
                                    if (startFails) throw new AssertionError("start failed");
                                    return new ServerCall.Listener<>() {};
                                });
            } catch (AssertionError expected) {
                assertTrue(startFails);
            }
            SerializingExecutor serial = new SerializingExecutor(e);
            serial.execute(
                    () -> {
                        throw new AssertionError("callback Error");
                    });
            serial.execute(retained::release);
            if (!startFails) {
                final var terminal = listener;
                serial.execute(terminal::onCancel);
            }
            tracer.streamClosed(Status.CANCELLED);
            assertEquals(1, guard.receiving());
            assertEquals(65536, allocator.pinnedDirectMemory());
            assertThrows(AssertionError.class, () -> work.remove().run());
            // Error has escaped real grpc-core SerializingExecutor, but cleanup is still queued.
            assertEquals(1, guard.receiving());
            assertEquals(65536, allocator.pinnedDirectMemory());
            int n = 0;
            while (!work.isEmpty()) {
                work.remove().run();
                assertTrue(++n < 10);
            }
            assertEquals(0, guard.receiving());
            assertEquals(0, allocator.pinnedDirectMemory());
            assertEquals(0, guard.pendingTimers());
            System.out.println(
                    "REAL_BYTEBUF_ERROR startFails="
                            + startFails
                            + " receiving="
                            + guard.receiving()
                            + " pinned="
                            + allocator.pinnedDirectMemory());
        } finally {
            if (retained.refCnt() > 0) retained.release();
            c.detach(before);
            guard.close();
        }
    }

    @Test
    void terminalBeforeSupplierDoesNotCreateOrphanWatchdog() {
        var guard = new GrpcReceivingGuard(1, 60000, Runnable::run);
        var tracer = guard.newServerStreamTracer("test/Schedule", new Metadata());
        Context c = tracer.filterContext(Context.ROOT), before = c.attach();
        try {
            tracer.streamClosed(Status.CANCELLED);
            guard.getExecutor(new Call(), new Metadata());
            assertEquals(0, guard.receiving());
            assertEquals(0, guard.pendingTimers());
        } finally {
            c.detach(before);
            guard.close();
        }
    }

    @Test
    void shutdownClosesReceivingCallsAndTerminatesWatchdog() throws Exception {
        var guard = new GrpcReceivingGuard(1, 60000, Runnable::run);
        var tracer = guard.newServerStreamTracer("test/Schedule", new Metadata());
        Context c = tracer.filterContext(Context.ROOT), before = c.attach();
        try {
            guard.getExecutor(
                    new Call() {
                        @Override
                        public void close(Status s, Metadata h) {
                            tracer.streamClosed(s);
                        }
                    },
                    new Metadata());
            assertEquals(1, guard.pendingTimers());
            guard.close();
            assertEquals(0, guard.receiving());
            assertEquals(0, guard.pendingTimers());
            var f = GrpcReceivingGuard.class.getDeclaredField("timer");
            f.setAccessible(true);
            assertTrue(((ScheduledThreadPoolExecutor) f.get(guard)).isTerminated());
        } finally {
            c.detach(before);
            guard.close();
        }
    }
}
