package org.flexlb.mock.vit;

import io.grpc.Status;
import java.util.*;
import java.net.http.HttpClient;
import java.time.Duration;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.MultimodalInputPB;

/** Native async-media workers. The independent scheduler owns the only Forward lane. */
final class VitPreparation implements AutoCloseable {
    private final VitConfig config;
    private final VitCache cache;
    private final VitScheduler scheduler;
    private final VitMetrics metrics;
    private final ThreadPoolExecutor workers;
    private final HttpClient client=HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5))
        .followRedirects(HttpClient.Redirect.NORMAL).build();
    private final Set<Task> tasks=ConcurrentHashMap.newKeySet();
    private volatile boolean closed;

    VitPreparation(VitConfig config,VitCache cache,VitScheduler scheduler,VitMetrics metrics) {
        this.config=config; this.cache=cache; this.scheduler=scheduler; this.metrics=metrics;
        // Cache admission already bounds queued + active unique media. A second
        // queue limit here would reject during worker completion/handoff races.
        workers=new ThreadPoolExecutor(config.concurrency(),config.concurrency(),0,TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(),Thread.ofVirtual().name("vit-prepare-"+config.id()+"-",0).factory());
    }

    void submit(VitCache.Claim claim,MultimodalInputPB media,long deadline) {
        if(closed) throw Status.UNAVAILABLE.withDescription("V preparation stopped").asRuntimeException();
        var task=new Task(claim,media,deadline);
        tasks.add(task); claim.onAbandon(task::cancel);
        try { workers.execute(task); }
        catch(RejectedExecutionException error) {
            tasks.remove(task); cache.complete(claim,Status.UNAVAILABLE.withDescription("V preparation stopped").asRuntimeException());
            throw Status.UNAVAILABLE.withCause(error).asRuntimeException();
        }
    }

    private final class Task implements Runnable {
        private final VitCache.Claim claim;
        private final MultimodalInputPB media;
        private final long deadline;
        private final VitImageSource source=new VitImageSource(config,metrics,client);
        private Thread runner;
        private boolean forwarded, cancelled;

        Task(VitCache.Claim claim,MultimodalInputPB media,long deadline) { this.claim=claim; this.media=media; this.deadline=deadline; }
        @Override public void run() {
            synchronized(this) { runner=Thread.currentThread(); }
            try {
                check();
                claim.prepare(source.prepare(media,deadline));
                synchronized(this) {
                    check(); scheduler.submit(claim,deadline); forwarded=true;
                }
                // A worker remains occupied while its GPU work is outstanding,
                // as in real _run_async_compute. Forward never waits on a worker.
                claim.result().get();
            } catch(InterruptedException error) {
                Thread.currentThread().interrupt(); cache.complete(claim,Status.CANCELLED.withDescription("V preparation interrupted").asRuntimeException());
            } catch(ExecutionException error) { /* Scheduler already published the failure. */ }
            catch(Exception error) { cache.complete(claim,error); }
            finally {
                synchronized(this) { runner=null; }
                source.close(); tasks.remove(this);
            }
        }
        private void check() {
            if(closed || cancelled || claim.users().get()==0) throw Status.CANCELLED.withDescription("No V consumers remain").asRuntimeException();
            if(System.nanoTime()>=deadline) throw Status.DEADLINE_EXCEEDED.withDescription("V queue wait expired").asRuntimeException();
        }
        void cancel() {
            synchronized(this) {
                if(forwarded) {
                    if(scheduler.cancel(claim)) cache.complete(claim,Status.CANCELLED.withDescription("V ready work cancelled").asRuntimeException());
                    return; // Selected/running GPU work retains ownership until completion.
                }
                cancelled=true;
                if(runner!=null) runner.interrupt();
            }
            source.close();
            if(workers.remove(this)) {
                cache.complete(claim,Status.CANCELLED.withDescription("V queued work cancelled").asRuntimeException()); tasks.remove(this);
            }
        }
    }

    @Override public void close() {
        closed=true;
        for(var task:tasks) task.cancel();
        client.shutdownNow();
        for(var queued:workers.shutdownNow()) {
            var task=(Task)queued;
            cache.complete(task.claim,Status.UNAVAILABLE.withDescription("V preparation stopped").asRuntimeException()); tasks.remove(task);
        }
    }
}
