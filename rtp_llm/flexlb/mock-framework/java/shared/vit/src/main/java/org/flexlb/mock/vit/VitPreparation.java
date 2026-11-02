package org.flexlb.mock.vit;

import io.grpc.Status;
import java.util.*;
import java.net.http.HttpClient;
import java.time.Duration;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.multimodal.MediaInput;

/** Owns media admission and terminal publication; Scheduler only completes Forward work. */
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
        // The task set bounds queued + active media, including result publication.
        workers=new ThreadPoolExecutor(config.concurrency(),config.concurrency(),0,TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(),Thread.ofVirtual().name("vit-prepare-"+config.id()+"-",0).factory());
    }

    synchronized List<VitCache.Claim> submit(MultimodalInputsPB input) {
        if(closed) throw Status.UNAVAILABLE.withDescription("V preparation stopped").asRuntimeException();
        VitGeometry.validate(input,config.skipDownload());
        var media=input.getMultimodalInputsList();
        var claims=cache.acquire(media.stream().map(MediaInput::key).toList());
        long count=claims.stream().filter(VitCache.Claim::producer).count();
        if(tasks.size()+count>config.queueSize()) {
            var error=Status.RESOURCE_EXHAUSTED.withDescription("V unique media admission full").asRuntimeException();
            claims.stream().filter(VitCache.Claim::producer).forEach(claim->cache.publish(claim,null,error));
            claims.forEach(VitCache::release); throw error;
        }
        long timeout=input.getTimeoutMs()>0?input.getTimeoutMs():media.stream()
            .mapToInt(value->value.getMmPreprocessConfig().getMmTimeoutMs()).filter(value->value>0).max().orElse(config.timeoutMs());
        long deadline=System.nanoTime()+TimeUnit.MILLISECONDS.toNanos(timeout);
        RuntimeException rejected=null;
        for(int i=0;i<claims.size();i++) if(claims.get(i).producer()) {
            var task=new Task(claims.get(i),media.get(i),deadline);
            tasks.add(task); task.claim.onAbandon(task::cancel);
            try { workers.execute(task); }
            catch(RejectedExecutionException error) {
                rejected=Status.UNAVAILABLE.withDescription("V preparation stopped").withCause(error).asRuntimeException();
                task.finish(null,rejected);
            }
        }
        if(rejected!=null) { claims.forEach(VitCache::release); throw rejected; }
        return claims;
    }

    private final class Task implements Runnable {
        private final VitCache.Claim claim;
        private final MultimodalInputPB media;
        private final long deadline;
        private final VitImageSource source=new VitImageSource(config,metrics,client);
        private Thread runner;
        private VitScheduler.Work work;
        private boolean cancelled,finished,prepared;

        Task(VitCache.Claim claim,MultimodalInputPB media,long deadline) { this.claim=claim; this.media=media; this.deadline=deadline; }
        @Override public void run() {
            synchronized(this) { if(finished) return; runner=Thread.currentThread(); }
            VitGeometry.Image result=null; Throwable failure=null;
            try {
                check();
                var image=source.prepare(media,deadline);
                synchronized(this) { check(); prepared=true; work=scheduler.submit(image,deadline); }
                result=work.result().get();
            } catch(InterruptedException error) {
                Thread.currentThread().interrupt(); failure=Status.CANCELLED.withDescription("V preparation interrupted").asRuntimeException();
            } catch(ExecutionException error) { failure=error.getCause(); }
            catch(Exception error) { failure=error; }
            finally { finish(result,failure); }
        }
        private void check() {
            if(closed || cancelled || claim.users().get()==0) throw Status.CANCELLED.withDescription("No V consumers remain").asRuntimeException();
            if(System.nanoTime()>=deadline) throw Status.DEADLINE_EXCEEDED.withDescription("V queue wait expired").asRuntimeException();
        }
        void finish(VitGeometry.Image image,Throwable error) {
            synchronized(this) { if(finished) return; finished=true; runner=null; }
            try { source.close(); cache.publish(claim,image,error); }
            finally { tasks.remove(this); } // Admission is released only after publication.
        }
        void cancel() {
            synchronized(this) {
                if(finished) return;
                if(work!=null) { scheduler.cancel(work); return; }
                cancelled=true;
                if(runner!=null) runner.interrupt();
            }
            source.close();
            if(workers.remove(this)) finish(null,Status.CANCELLED.withDescription("V queued work cancelled").asRuntimeException());
        }
    }

    int inflight() { return tasks.size(); }
    int preparing() {
        return (int)tasks.stream().filter(task->{ synchronized(task) { return !task.finished && !task.prepared; } }).count();
    }
    @Override public void close() {
        synchronized(this) { closed=true; }
        for(var task:tasks) task.cancel();
        client.shutdownNow();
        for(var queued:workers.shutdownNow()) ((Task)queued).finish(null,Status.UNAVAILABLE.withDescription("V preparation stopped").asRuntimeException());
    }
}
