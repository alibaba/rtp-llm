package org.flexlb.mock.client;

import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleRequestPB;
import org.flexlb.mock.client.ClientProtocol.*;

/** Owns admission, concurrency and request lifetimes. Only Player controls arrival rate. */
final class LoadClient implements AutoCloseable {
    private final Shard plan;
    private final NativeClient client;
    private final MasterDiscovery discovery;
    private final ClientMetrics metrics;
    private final ArrayBlockingQueue<Item> queue;
    private final Map<Long, NativeClient.Call> active = new HashMap<>();
    private final Map<String, Long> outcomes = new TreeMap<>();
    private final List<String> errors = new ArrayList<>();
    private final CompletableFuture<State> completion = new CompletableFuture<>();
    private Thread player, dispatcher;
    private boolean begun, producerDone;
    private volatile boolean stopped;
    private final AtomicLong produced = new AtomicLong();
    private long issued, finished, drainDeadline = Long.MAX_VALUE;

    LoadClient(Shard plan, ClientMetrics metrics, java.nio.file.Path topology) throws Exception {
        this.plan = plan; this.metrics = metrics;
        queue = new ArrayBlockingQueue<>(plan.queue_capacity());
        for (var outcome : NativeClient.Outcome.values()) outcomes.put(outcome.name(), 0L);
        discovery = new MasterDiscovery(topology, plan.masters());
        client = new NativeClient(discovery::routes);
        // A lost begin command must not leave a prepared worker alive forever.
        dispatcher = Thread.ofVirtual().name("client-dispatch").start(this::dispatch);
    }
    synchronized void begin(long epochMillis) {
        if (begun || stopped || completion.isDone()) throw new IllegalStateException("Client already begun or stopped");
        long delay = epochMillis - System.currentTimeMillis();
        if (delay < 0 || delay > 30000) throw new IllegalArgumentException("Start time must be within the next 30 seconds");
        begun = true;
        long start = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(delay);
        player = Thread.ofVirtual().name("client-player").start(() -> {
            try { Player.play(plan.requests(), plan.qps(), start, this::offer); }
            catch (InterruptedException interrupted) { if (!isStopped()) fail("Player interrupted unexpectedly"); }
            catch (RuntimeException error) { fail(error.toString()); }
            finally { synchronized (LoadClient.this) { finishInput(); } }
        });
        LockSupport.unpark(dispatcher);
    }
    private synchronized boolean isStopped() { return stopped; }
    private void offer(Item item) {
        if (stopped) return;
        if (!queue.offer(item)) {
            synchronized (this) { fail("Playback queue full: capacity=" + plan.queue_capacity() + ", inflight=" + active.size()); }
            throw new IllegalStateException("Playback queue full");
        }
        produced.incrementAndGet();
        // Stop may race an offer; it can never reopen this queue or issue that request.
        if (stopped) queue.clear();
        LockSupport.unpark(dispatcher);
    }
    private void finishInput() {
        if (!producerDone) {
            producerDone = true;
            drainDeadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(plan.drain_timeout_ms());
        }
        LockSupport.unpark(dispatcher);
    }
    synchronized void halt() {
        if (completion.isDone() || stopped) return;
        stopped = true; queue.clear();
        if (player != null) player.interrupt();
        finishInput();
    }
    synchronized void fail(String error) {
        if (errors.isEmpty()) errors.add(error);
        halt();
    }
    private void dispatch() {
        long prepareDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        try {
            while (true) {
                discovery.check();
                List<NativeClient.Call> cancel = List.of();
                synchronized (this) {
                    if (!begun && !stopped && System.nanoTime() >= prepareDeadline) fail("Timed out waiting for begin");
                    if (producerDone && queue.isEmpty() && active.isEmpty()) break;
                    if (producerDone && System.nanoTime() >= drainDeadline) {
                        fail("Client drain timed out");
                        cancel = List.copyOf(active.values());
                    } else if (begun && !stopped && !queue.isEmpty() && active.size() < plan.concurrency()) {
                        var item = queue.remove();
                        var input = GenerateInputPB.newBuilder().setRequestId(item.request_id()).addAllTokenIds(item.input().token_ids())
                            .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(item.input().max_new_tokens())
                                .setCanUsePdSeparation(true).setReturnIncremental(true)).build();
                        var request = FlexlbScheduleRequestPB.newBuilder().setRequestId(item.request_id())
                            .setSeqLen(item.input().token_ids().size()).setMaxNewTokens(item.input().max_new_tokens())
                            .setPriority(item.input().priority()).setGenerateInput(input.toByteString()).build();
                        var call = client.send(request, Duration.ofMillis(plan.request_timeout_ms()));
                        active.put(item.request_id(), call); issued++; metrics.issued();
                        call.completion().whenComplete((result, error) -> {
                            synchronized (LoadClient.this) {
                                active.remove(item.request_id()); finished++;
                                if (error != null) fail("Request execution failed: " + error);
                                else { metrics.finished(result); outcomes.merge(result.outcome().name(), 1L, Long::sum); }
                                LockSupport.unpark(dispatcher);
                            }
                        });
                        continue;
                    }
                }
                // Native completion callbacks acquire this object's monitor; never cancel while holding it.
                cancel.forEach(NativeClient.Call::cancel);
                LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(100));
                if (Thread.interrupted()) throw new InterruptedException();
            }
        } catch (InterruptedException error) {
            fail("Client dispatcher interrupted"); Thread.currentThread().interrupt();
        } catch (RuntimeException error) { fail("Client dispatcher failed: " + error); }
        finally {
            if (player != null) try { player.join(1000); }
            catch (InterruptedException error) { Thread.currentThread().interrupt(); }
            client.close(); discovery.close();
            synchronized (this) { completion.complete(snapshot(true)); LockSupport.unpark(dispatcher); }
        }
    }
    private State snapshot(boolean done) {
        return new State(done, stopped, produced.get(), issued, finished, plan.requests().size() - issued,
            queue.size(), active.size(), Map.copyOf(outcomes), List.copyOf(errors));
    }
    synchronized State state() { return completion.isDone() ? completion.join() : snapshot(false); }
    State await(long timeoutMillis) throws Exception { return completion.get(timeoutMillis, TimeUnit.MILLISECONDS); }
    @Override public void close() {
        halt();
        List<NativeClient.Call> calls;
        synchronized (this) { calls = List.copyOf(active.values()); }
        calls.forEach(NativeClient.Call::cancel);
        try { completion.get(8, TimeUnit.SECONDS); }
        catch (Exception error) { dispatcher.interrupt(); }
    }
}
