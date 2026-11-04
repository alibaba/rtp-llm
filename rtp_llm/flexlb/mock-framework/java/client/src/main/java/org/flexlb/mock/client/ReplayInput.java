package org.flexlb.mock.client;

import com.github.luben.zstd.*;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.security.*;
import java.util.*;
import org.flexlb.mock.client.input.ClientInput.*;
import org.flexlb.mock.client.ClientProtocol.Start;

/** Versioned, bounded Protobuf chunks; source files and prepared shards use this same codec. */
final class ReplayInput {
    private ReplayInput() {}

    static final byte[] MAGIC = "MCKINP01".getBytes(StandardCharsets.US_ASCII);
    static final int MAX_FRAME = 4 * 1024 * 1024, MAX_REQUEST = 1024 * 1024, MAX_CHUNK_REQUESTS = 1024;

    static void validate(Request request) throws IOException {
        if (request.getTokenIdsCount() == 0 || request.getTokenIdsList().stream().anyMatch(v -> v < 0)
                || request.getMaxNewTokens() < 1 || request.hasArrivalOffsetMs() && request.getArrivalOffsetMs() < 0
                || request.hasPriority() && request.getPriority() < 0
                || request.getSerializedSize() > MAX_REQUEST || !request.getUnknownFields().asMap().isEmpty())
            throw new IOException("Invalid or unsupported playback request");
        int end = 0;
        if (request.getImagesCount() > 256) throw new IOException("Too many images");
        for (var image : request.getImagesList()) {
            if (image.getUrl().isBlank() || image.getUrl().length() > 4096 || image.getWidth() < 1 || image.getWidth() > 32768
                    || image.getHeight() < 1 || image.getHeight() > 32768 || image.getOffset() < end || image.getLength() < 1
                    || (long) image.getOffset() + image.getLength() > request.getTokenIdsCount() || !image.getUnknownFields().asMap().isEmpty())
                throw new IOException("Invalid expanded image span or geometry");
            end = image.getOffset() + image.getLength();
        }
    }
    static void checkInterrupted() throws InterruptedIOException {
        if (Thread.currentThread().isInterrupted()) throw new InterruptedIOException("Request preparation/playback interrupted");
    }
    static MessageDigest digest() {
        try { return MessageDigest.getInstance("SHA-256"); }
        catch (NoSuchAlgorithmException error) { throw new AssertionError(error); }
    }
    static final class Writer implements AutoCloseable {
        private final DataOutputStream stream;
        private final InputChunk.Builder chunk = InputChunk.newBuilder();
        private long sequence, count;
        private int bytes;
        private boolean finished;
        Writer(OutputStream stream) throws IOException {
            this.stream = new DataOutputStream(stream);
            try { this.stream.write(MAGIC); }
            catch (IOException error) {
                try { this.stream.close(); } catch (IOException close) { error.addSuppressed(close); }
                throw error;
            }
        }
        void write(Request request) throws IOException {
            checkInterrupted();
            if (finished) throw new IOException("Input writer is already sealed");
            validate(request);
            int size = request.getSerializedSize() + 5;
            if (chunk.getRequestsCount() > 0 && (bytes + size > MAX_REQUEST || chunk.getRequestsCount() == MAX_CHUNK_REQUESTS)) flush();
            chunk.addRequests(request); bytes += size; count = Math.addExact(count, 1);
        }
        private void flush() throws IOException {
            chunk.setSequence(++sequence);
            try (var compressor = new ZstdCompressCtx().setLevel(3).setChecksum(true).setContentSize(true)) {
                byte[] encoded = compressor.compress(chunk.build().toByteArray());
                if (encoded.length > MAX_FRAME) throw new IOException("Oversized compressed input chunk");
                stream.writeInt(encoded.length); stream.write(encoded);
            } catch (ZstdException error) { throw new IOException("Cannot compress input chunk", error); }
            chunk.clear(); bytes = 0;
        }
        void finish() throws IOException {
            if (finished) throw new IOException("Input writer is already sealed");
            if (chunk.getRequestsCount() > 0) flush();
            chunk.setTotalRequests(count); flush(); finished = true;
            stream.flush();
        }
        @Override public void close() throws IOException { stream.close(); }
    }
    static final class Reader implements AutoCloseable {
        private final Path path;
        private final MessageDigest digest = digest();
        private final DataInputStream stream;
        private Iterator<Request> requests = Collections.emptyIterator();
        private long sequence, count;
        private boolean finished;
        Reader(Path path) throws IOException {
            this.path = path;
            if (!Files.isRegularFile(path)) throw new IOException("Not a regular input file: " + path);
            stream = new DataInputStream(new BufferedInputStream(new DigestInputStream(Files.newInputStream(path), digest)));
            try {
                if (!Arrays.equals(stream.readNBytes(MAGIC.length), MAGIC)) throw new IOException("Expected Protobuf+Zstd playback input v1");
            } catch (IOException error) {
                try { stream.close(); } catch (IOException close) { error.addSuppressed(close); }
                throw error;
            }
        }
        Request next() throws IOException {
            checkInterrupted();
            if (finished) return null;
            if (!requests.hasNext()) {
                long expectedSequence = sequence + 1;
                try {
                    int size = stream.readInt();
                    if (size < 6 || size > MAX_FRAME) throw new IOException("Invalid compressed input chunk size");
                    byte[] encoded = stream.readNBytes(size);
                    if (encoded.length != size) throw new EOFException("Truncated input chunk");
                    long decodedSize = Zstd.getFrameContentSize(encoded);
                    if (decodedSize < 1 || decodedSize > MAX_FRAME || (encoded[4] & 0x24) != 0x24)
                        throw new IOException("Input chunk requires bounded content size and checksum");
                    var chunk = decode(encoded, (int) decodedSize);
                    if (!chunk.getUnknownFields().asMap().isEmpty() || chunk.getSequence() != expectedSequence)
                        throw new IOException("Unknown input fields or noncontiguous chunk sequence");
                    sequence = expectedSequence;
                    if (chunk.hasTotalRequests()) {
                        if (chunk.getRequestsCount() != 0 || chunk.getTotalRequests() != count || stream.read() != -1)
                            throw new IOException("Input seal count mismatch or trailing data");
                        finished = true; return null;
                    }
                    if (chunk.getRequestsCount() < 1 || chunk.getRequestsCount() > MAX_CHUNK_REQUESTS)
                        throw new IOException("Invalid input chunk request count");
                    // Validate the whole chunk before exposing any of its requests.
                    for (var request : chunk.getRequestsList()) validate(request);
                    requests = chunk.getRequestsList().iterator();
                } catch (IOException | ZstdException error) {
                    throw new IOException(path + ": chunk " + expectedSequence + ": " + error.getMessage(), error);
                }
            }
            count = Math.addExact(count, 1);
            return requests.next();
        }
        private static InputChunk decode(byte[] encoded, int size) throws IOException {
            var source = ByteBuffer.allocateDirect(encoded.length);
            source.put(encoded).flip();
            var target = ByteBuffer.allocateDirect(size + 1);
            try (var decoder = new ZstdDecompressCtx()) {
                if (!decoder.decompressDirectByteBufferStream(target, source) || source.hasRemaining() || target.position() != size)
                    throw new IOException("Input chunk must contain exactly one complete Zstd frame");
                return InputChunk.parseFrom(target.flip());
            }
        }
        void verify(String expected) throws IOException {
            if (!finished || !HexFormat.of().formatHex(digest.digest()).equals(expected))
                throw new IOException("Incomplete input or shard checksum mismatch: " + path);
        }
        @Override public void close() throws IOException { stream.close(); }
    }

    static long delayNanos(long index, org.flexlb.mock.client.input.ClientInput.Request input, Double qps, Double speed) throws IOException {
        if (speed != null && !input.hasArrivalOffsetMs()) throw new IOException("Replay requires arrival_offset_ms on every request");
        double seconds = speed == null ? index / qps : input.getArrivalOffsetMs() / (1000.0 * speed);
        if (!Double.isFinite(seconds) || seconds < 0 || seconds > 86400) throw new IOException("Playback exceeds 24 hours");
        return (long) (seconds * 1_000_000_000.0);
    }
    record Part(String path, long count, String sha256) {}
    static List<Part> split(Start plan, Path directory, int clients, java.util.function.BooleanSupplier cancelled) throws IOException {
        Files.createDirectory(directory);
        var outputs = new ArrayList<ReplayInput.Writer>();
        var digests = new ArrayList<MessageDigest>();
        long count = 0, previousArrival = -1;
        Throwable primary = null;
        try {
            for (int i = 0; i < clients; i++) {
                var digest = ReplayInput.digest(); digests.add(digest);
                outputs.add(new ReplayInput.Writer(new BufferedOutputStream(new DigestOutputStream(Files.newOutputStream(
                    directory.resolve(i + ".part"), StandardOpenOption.CREATE_NEW), digest))));
            }
            Path source = Path.of(plan.requests_file());
            var before = Files.readAttributes(source, BasicFileAttributes.class);
            try (var reader = new ReplayInput.Reader(source)) {
                org.flexlb.mock.client.input.ClientInput.Request input;
                while ((input = reader.next()) != null) {
                    ReplayInput.checkInterrupted();
                    if (cancelled.getAsBoolean()) throw new InterruptedIOException("Request preparation stopped");
                    long next = Math.addExact(count, 1);
                    if (plan.qps() != null && next / plan.qps() > 86400) throw new IOException("Playback exceeds 24 hours");
                    ReplayInput.delayNanos(count, input, plan.qps(), plan.replay_speed());
                    if (plan.replay_speed() != null) {
                        if (input.getArrivalOffsetMs() < previousArrival) throw new IOException("Replay arrival offsets must be nondecreasing");
                        previousArrival = input.getArrivalOffsetMs();
                    }
                    outputs.get((int) (count % clients)).write(input);
                    count = next;
                }
            }
            var after = Files.readAttributes(source, BasicFileAttributes.class);
            if (before.size() != after.size() || !before.lastModifiedTime().equals(after.lastModifiedTime())
                    || !Objects.equals(before.fileKey(), after.fileKey())) throw new IOException("Input file changed during preparation: " + source);
            if (count < clients) throw new IOException("Need at least one request per Client");
            for (var output : outputs) output.finish();
        } catch (IOException | RuntimeException | Error failure) {
            primary = failure; throw failure;
        } finally {
            IOException failure = null;
            for (var output : outputs) try { output.close(); }
            catch (IOException error) { if (failure == null) failure = error; else failure.addSuppressed(error); }
            if (failure != null) {
                if (primary != null) primary.addSuppressed(failure);
                else throw failure;
            }
        }
        var parts = new ArrayList<Part>();
        for (int i = 0; i < clients; i++) {
            ReplayInput.checkInterrupted();
            Path path = directory.resolve(i + ".binpb.zst").toAbsolutePath();
            Files.move(directory.resolve(i + ".part"), path, StandardCopyOption.ATOMIC_MOVE);
            parts.add(new Part(path.toString(), count / clients + (i < count % clients ? 1 : 0), HexFormat.of().formatHex(digests.get(i).digest())));
        }
        return List.copyOf(parts);
    }
}
