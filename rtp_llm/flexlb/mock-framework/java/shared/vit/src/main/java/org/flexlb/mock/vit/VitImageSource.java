package org.flexlb.mock.vit;

import io.grpc.Status;
import java.io.*;
import java.net.*;
import java.net.http.*;
import java.time.Duration;
import java.util.concurrent.*;
import java.nio.*;
import java.security.*;
import java.util.HexFormat;
import javax.imageio.ImageIO;
import javax.imageio.stream.MemoryCacheImageInputStream;
import org.flexlb.engine.grpc.EngineRpcService.MultimodalInputPB;
import org.flexlb.mock.multimodal.MediaInput;

/** One cancellable image read. No pixel tensor is decoded or retained. */
final class VitImageSource implements AutoCloseable {
    private final VitConfig config;
    private final VitMetrics metrics;
    private final HttpClient client;
    private volatile CompletableFuture<?> request;
    private volatile boolean closed;

    VitImageSource(VitConfig config, VitMetrics metrics, HttpClient client) { this.config=config; this.metrics=metrics; this.client=client; }

    VitGeometry.Image prepare(MultimodalInputPB media, long deadline) throws IOException {
        int width, height;
        String identity=MediaInput.key(media);
        if(config.skipDownload()) {
            width=media.getMmPreprocessConfig().getWidth(); height=media.getMmPreprocessConfig().getHeight();
        } else {
            long started=System.nanoTime();
            byte[] bytes;
            try { bytes=download(media.getMultimodalUrl(),Math.min(deadline,System.nanoTime()+TimeUnit.SECONDS.toNanos(60))); }
            finally {
                double millis=(System.nanoTime()-started)/1e6;
                metrics.events.observe("py_rtp_vit_download_rt",millis);
                metrics.events.observe("rtp_llm_vit_image_fetch_rt_us",millis*1000);
            }
            int[] size=dimensions(bytes); width=size[0]; height=size[1];
            identity=contentIdentity(bytes);
        }
        VitGeometry.dimensions(width,height);
        return new VitGeometry.Image(MediaInput.key(media),width,height,
            VitGeometry.rows(width,height,config.patch(),config.merge()),identity);
    }

    private String contentIdentity(byte[] bytes) {
        try {
            String digest=HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
            return digest+":"+config.patch()+":"+config.merge()+":"+config.hidden();
        } catch(NoSuchAlgorithmException error) { throw new AssertionError(error); }
    }

    private byte[] download(String url,long deadline) {
        URI uri;
        try { uri=URI.create(url); }
        catch(IllegalArgumentException error) { throw MediaInput.invalid("Invalid image URL"); }
        if(uri.getHost()==null || (!"http".equalsIgnoreCase(uri.getScheme()) && !"https".equalsIgnoreCase(uri.getScheme())))
            throw MediaInput.invalid("Image download requires an HTTP(S) URL");
        final long maxBytes=config.maxFileSizeKb()>0 ? config.maxFileSizeKb()*1024L : Integer.MAX_VALUE-8L;
        for(int attempt=0; ; attempt++) {
            check(deadline);
            var message=HttpRequest.newBuilder(uri).timeout(Duration.ofNanos(Math.max(1,deadline-System.nanoTime())))
                .header("User-Agent","RTP-LLM-Mock/1.0");
            config.downloadHeaders().forEach(message::setHeader);
            var output=new ByteArrayOutputStream();
            var future=client.sendAsync(message.GET().build(),info->{
                if(info.headers().firstValueAsLong("Content-Length").orElse(-1)>maxBytes)
                    throw MediaInput.invalid("Image exceeds download byte limit");
                return HttpResponse.BodySubscribers.ofByteArrayConsumer(chunk->chunk.ifPresent(bytes->{
                    if(output.size()+(long)bytes.length>maxBytes) throw MediaInput.invalid("Image exceeds download byte limit");
                    output.writeBytes(bytes);
                }));
            });
            request=future;
            try {
                check(deadline);
                var response=future.get(Math.max(1,deadline-System.nanoTime()),TimeUnit.NANOSECONDS);
                int status=response.statusCode();
                if(status!=200) {
                    if(attempt<2 && (status==408 || status==429 || status==500 || status==502 || status==503 || status==504)) continue;
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
            } finally { future.cancel(true); request=null; }
        }
    }

    private void check(long deadline) {
        if(closed || Thread.currentThread().isInterrupted()) throw Status.CANCELLED.withDescription("Image preparation cancelled").asRuntimeException();
        if(System.nanoTime()>=deadline) throw Status.DEADLINE_EXCEEDED.withDescription("Image preparation expired").asRuntimeException();
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
            if(!readers.hasNext()) throw MediaInput.invalid("Unsupported or malformed image header");
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

    @Override public void close() { closed=true; var active=request; if(active!=null) active.cancel(true); }
}
