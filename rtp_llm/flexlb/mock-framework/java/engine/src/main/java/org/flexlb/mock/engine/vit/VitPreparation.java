package org.flexlb.mock.engine.vit;

import io.grpc.Status;
import java.io.*;
import java.net.URI;
import java.net.http.*;
import java.nio.*;
import java.security.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import javax.imageio.ImageIO;
import javax.imageio.stream.MemoryCacheImageInputStream;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.protocol.MultimodalProtocol;

/**
 * Owns media admission, prepares inputs and publishes results after Forward completes.
 */
final class VitPreparation implements AutoCloseable {
    private final VitConfig config;
    private final VitCache cache;
    private final VitScheduler scheduler;
    private final VitMetrics metrics;
    private final ThreadPoolExecutor workers;
    private final HttpClient client=HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5))
        .followRedirects(HttpClient.Redirect.ALWAYS).build();
    private final Set<Task> tasks=ConcurrentHashMap.newKeySet();
    private final java.util.concurrent.atomic.AtomicInteger preparing = new java.util.concurrent.atomic.AtomicInteger();

    private volatile boolean closed;

    VitPreparation(VitConfig config,VitCache cache,VitScheduler scheduler,VitMetrics metrics) {
        this.config=config; this.cache=cache; this.scheduler=scheduler; this.metrics=metrics;
        // The task set bounds queued + active media, including result publication.
        workers=new ThreadPoolExecutor(config.concurrency(),config.concurrency(),0,TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(),Thread.ofVirtual().name("vit-prepare-"+config.id()+"-",0).factory());
    }

    @Override public void close() {
        synchronized(this) { closed=true; }
        for(var task:tasks) task.cancel();
        client.shutdownNow();
        for(var queued:workers.shutdownNow()) ((Task)queued).finish(null,Status.UNAVAILABLE.withDescription("V preparation stopped").asRuntimeException());
    }

    /** Admits inputs already validated at the RPC boundary. */
    synchronized List<VitCache.Claim> submit(MultimodalInputsPB input) {
        if(closed) throw Status.UNAVAILABLE.withDescription("V preparation stopped").asRuntimeException();
        var media=input.getMultimodalInputsList();
        var claims=cache.acquire(media.stream().map(MultimodalProtocol::key).toList());
        long count=claims.stream().filter(VitCache.Claim::producer).count();
        if(tasks.size()+count>config.queueSize()) {
            var error=Status.RESOURCE_EXHAUSTED.withDescription("V unique media admission full").asRuntimeException();
            claims.stream().filter(VitCache.Claim::producer).forEach(claim->cache.publish(claim,null,error));
            claims.forEach(VitCache.Claim::close); throw error;
        }
        long timeout=input.getTimeoutMs()>0?input.getTimeoutMs():media.stream()
            .mapToInt(value->value.getMmPreprocessConfig().getMmTimeoutMs()).filter(value->value>0).max().orElse(config.timeoutMs());
        long deadline=System.nanoTime()+TimeUnit.MILLISECONDS.toNanos(timeout);
        RuntimeException rejected=null;
        for(int i=0;i<claims.size();i++) if(claims.get(i).producer()) {
            var task=new Task(claims.get(i),media.get(i),deadline);
            tasks.add(task);
            metrics.events.observe("py_rtp_vit_preprocess_queue_size", preparing.incrementAndGet());
            task.claim.onAbandon(task::cancel);
            try { workers.execute(task); }
            catch(RejectedExecutionException error) {
                rejected=Status.UNAVAILABLE.withDescription("V preparation stopped").withCause(error).asRuntimeException();
                task.finish(null,rejected);
            }
        }
        if(rejected!=null) { claims.forEach(VitCache.Claim::close); throw rejected; }
        return claims;
    }

    int inflight() { return tasks.size(); }

    private void leavePreparation() {
        metrics.events.observe("py_rtp_vit_preprocess_queue_size", preparing.decrementAndGet());
    }

    /**
     * Prepares one unique image and holds its admission slot until publication.
     */
    private final class Task implements Runnable {
        private final VitCache.Claim claim;
        private final MultimodalInputPB media;

        private final long deadline;

        private final ImageRead source = new ImageRead();

        private Thread runner;
        private VitScheduler.Work work;

        private boolean cancelled,finished,prepared;

        Task(VitCache.Claim claim,MultimodalInputPB media,long deadline) { this.claim=claim; this.media=media; this.deadline=deadline; }

        @Override public void run() {
            synchronized(this) { if(finished) return; runner=Thread.currentThread(); }
            VitGeometry.Image result=null; Throwable failure=null;
            try {
                check();
                metrics.started();
                var image=source.prepare(media,deadline);
                synchronized(this) {
                    check();
                    prepared=true;
                    leavePreparation();
                    work=scheduler.submit(image,deadline);
                }
                result=work.result().toCompletableFuture().get();
            } catch(InterruptedException error) {
                Thread.currentThread().interrupt(); failure=Status.CANCELLED.withDescription("V preparation interrupted").asRuntimeException();
            } catch(ExecutionException error) { failure=error.getCause(); }
            catch(Exception error) { failure=error; }
            finally { finish(result,failure); }
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

        void finish(VitGeometry.Image image,Throwable error) {
            synchronized(this) {
                if(finished) return;
                finished=true; runner=null;
                if(!prepared) leavePreparation();
            }
            try { source.close(); cache.publish(claim,image,error); }
            finally { tasks.remove(this); } // Admission is released only after publication.
        }

        private void check() {
            if(closed || cancelled || !claim.hasConsumers()) throw Status.CANCELLED.withDescription("No V consumers remain").asRuntimeException();
            if(System.nanoTime()>=deadline) throw Status.DEADLINE_EXCEEDED.withDescription("V queue wait expired").asRuntimeException();
        }
    }

    /**
     * Owns one cancellable image read and extracts dimensions without decoding pixels.
     */
    private final class ImageRead implements AutoCloseable {
        private volatile CompletableFuture<?> request;

        private volatile boolean closed;

        @Override public void close() { closed=true; var active=request; if(active!=null) active.cancel(true); }

        VitGeometry.Image prepare(MultimodalInputPB media, long deadline) throws IOException {
            int width, height;
            String requestKey=MultimodalProtocol.key(media);
            String identity=requestKey;
            long started=System.nanoTime(), downloadNanos=0;
            if(config.skipDownload()) {
                width=media.getMmPreprocessConfig().getWidth(); height=media.getMmPreprocessConfig().getHeight();
            } else {
                byte[] bytes=download(media.getMultimodalUrl(),Math.min(deadline,System.nanoTime()+TimeUnit.SECONDS.toNanos(60)));
                downloadNanos=System.nanoTime()-started;
                // This image planner mirrors real DeepSeek V4.1 preprocessing.
                metrics.events.observe("rtp_llm_vit_image_fetch_rt_us",downloadNanos/1e3,
                    java.util.Map.of("model","deepseek_v41","mm_type","image"));
                int[] size=dimensions(bytes); width=size[0]; height=size[1];
                identity=contentIdentity(bytes);
            }
            VitGeometry.dimensions(width,height);
            var image=new VitGeometry.Image(requestKey,width,height,
                VitGeometry.rows(width,height,config.patch(),config.merge(),config.minPixels(),config.maxTokens()),identity);
            if (!config.skipDownload()) {
                long elapsed=System.nanoTime()-started;
                metrics.events.observe("py_rtp_vit_preprocess_rt",elapsed/1e6);
                metrics.events.observe("py_rtp_vit_download_rt",downloadNanos/1e6);
                metrics.events.observe("py_rtp_vit_preprocess_other_rt",(elapsed-downloadNanos)/1e6);
            }
            return image;
        }

        private String contentIdentity(byte[] bytes) {
            try {
                String digest=HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
                return digest+":"+config.patch()+":"+config.merge()+":"+config.minPixels()+":"+config.maxTokens()+":"+config.hidden();
            } catch(NoSuchAlgorithmException error) { throw new AssertionError(error); }
        }

        private byte[] download(String url,long deadline) {
            URI uri;
            try { uri=URI.create(url); }
            catch(IllegalArgumentException error) { throw MultimodalProtocol.invalid("Invalid image URL"); }
            if(uri.getHost()==null || (!"http".equalsIgnoreCase(uri.getScheme()) && !"https".equalsIgnoreCase(uri.getScheme())))
                throw MultimodalProtocol.invalid("Image download requires an HTTP(S) URL");
            // DSV4.1 load_image_source caps original media at 10 MiB, even when the generic file limit is disabled.
            final long maxBytes=config.maxFileSizeKb()>0 ? Math.min(10L*1024*1024,config.maxFileSizeKb()*1024L) : 10L*1024*1024;
            for(int attempt=0; ; attempt++) {
                check(deadline);
                var message=HttpRequest.newBuilder(uri).timeout(Duration.ofNanos(Math.max(1,deadline-System.nanoTime())));
                if(config.downloadHeaders().isEmpty()) {
                    message.header("User-Agent","Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0");
                    message.header("Accept","text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7");
                }
                config.downloadHeaders().forEach(message::setHeader);
                var output=new ByteArrayOutputStream();
                var future=client.sendAsync(message.GET().build(),info->{
                    if(info.headers().firstValueAsLong("Content-Length").orElse(-1)>maxBytes)
                        throw MultimodalProtocol.invalid("Image exceeds download byte limit");
                    return HttpResponse.BodySubscribers.ofByteArrayConsumer(chunk->chunk.ifPresent(bytes->{
                        if(output.size()+(long)bytes.length>maxBytes) throw MultimodalProtocol.invalid("Image exceeds download byte limit");
                        output.writeBytes(bytes);
                    }));
                });
                request=future;
                try {
                    check(deadline);
                    var response=future.get(Math.max(1,deadline-System.nanoTime()),TimeUnit.NANOSECONDS);
                    int status=response.statusCode();
                    if(status!=200) {
                        if(attempt<2 && (status==408 || status==429 || status==500 || status==502 || status==503 || status==504)) {
                            retryPause(attempt,deadline); continue;
                        }
                        throw Status.UNAVAILABLE.withDescription("Image download HTTP " + status).asRuntimeException();
                    }
                    check(deadline); return output.toByteArray();
                } catch(TimeoutException error) {
                    throw Status.DEADLINE_EXCEEDED.withDescription("Image download expired").asRuntimeException();
                } catch(InterruptedException error) {
                    Thread.currentThread().interrupt(); throw Status.CANCELLED.withDescription("Image download interrupted").asRuntimeException();
                } catch(CancellationException error) {
                    throw Status.CANCELLED.withDescription("Image download cancelled").asRuntimeException();
                } catch(ExecutionException error) {
                    Throwable cause=error.getCause();
                    while(cause instanceof CompletionException && cause.getCause()!=null) cause=cause.getCause();
                    if(cause instanceof io.grpc.StatusRuntimeException status) throw status;
                    check(deadline);
                    if(cause instanceof HttpTimeoutException && !(cause instanceof HttpConnectTimeoutException)) throw Status.DEADLINE_EXCEEDED.withDescription("Image download expired").asRuntimeException();
                    if(attempt>=2) throw Status.UNAVAILABLE.withDescription("Image download failed").withCause(cause).asRuntimeException();
                    retryPause(attempt,deadline);
                } finally { future.cancel(true); request=null; }
            }
        }

        private void check(long deadline) {
            if(closed || Thread.currentThread().isInterrupted()) throw Status.CANCELLED.withDescription("Image preparation cancelled").asRuntimeException();
            if(System.nanoTime()>=deadline) throw Status.DEADLINE_EXCEEDED.withDescription("Image preparation expired").asRuntimeException();
        }

        private void retryPause(int attempt,long deadline) {
            check(deadline);
            try { TimeUnit.NANOSECONDS.sleep(Math.min(TimeUnit.MILLISECONDS.toNanos(200L << attempt),deadline-System.nanoTime())); }
            catch(InterruptedException error) { Thread.currentThread().interrupt(); }
            check(deadline);
        }

        private static int[] dimensions(byte[] bytes) throws IOException {
            // The JDK readers cover PNG/JPEG/GIF/BMP. WebP dimensions need no pixel codec.
            if(bytes.length>=25 && ascii(bytes,0,"RIFF") && ascii(bytes,8,"WEBP")) {
                if(bytes.length>=30 && ascii(bytes,12,"VP8X")) return new int[]{1+little24(bytes,24),1+little24(bytes,27)};
                if(ascii(bytes,12,"VP8L") && bytes[20]==0x2f) {
                    int bits=ByteBuffer.wrap(bytes,21,4).order(ByteOrder.LITTLE_ENDIAN).getInt();
                    return new int[]{1+(bits&0x3fff),1+((bits>>>14)&0x3fff)};
                }
                if(bytes.length>=30 && ascii(bytes,12,"VP8 ") && (bytes[23]&255)==0x9d && bytes[24]==1 && bytes[25]==0x2a)
                    return new int[]{((bytes[26]&255)|((bytes[27]&255)<<8))&0x3fff,((bytes[28]&255)|((bytes[29]&255)<<8))&0x3fff};
            }
            try(var input=new MemoryCacheImageInputStream(new ByteArrayInputStream(bytes))) {
                var readers=ImageIO.getImageReaders(input);
                if(!readers.hasNext()) throw MultimodalProtocol.invalid("Unsupported or malformed image header");
                var reader=readers.next();
                try { reader.setInput(input,true,true); return new int[]{reader.getWidth(0),reader.getHeight(0)}; }
                catch(IOException error) { throw Status.INVALID_ARGUMENT.withDescription("Malformed image header").withCause(error).asRuntimeException(); }
                finally { reader.dispose(); }
            }
        }

        private static boolean ascii(byte[] bytes,int offset,String text) {
            for(int i=0;i<text.length();i++) if(bytes[offset+i]!=(byte)text.charAt(i)) return false;
            return true;
        }

        private static int little24(byte[] bytes,int offset) { return (bytes[offset]&255)|((bytes[offset+1]&255)<<8)|((bytes[offset+2]&255)<<16); }
    }
}
