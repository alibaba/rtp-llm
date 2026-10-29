package org.flexlb.mock.client;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.*;
import java.io.*;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.security.*;
import java.util.*;
import org.flexlb.mock.client.ClientProtocol.*;

/** Prepare immutable local shards before playback; retain only one bounded record while reading. */
final class RequestFiles {
    static final int MAX_RECORD_BYTES = 1024 * 1024;
    private static final ObjectMapper JSON = new ObjectMapper()
        .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
        .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);
    record Part(String path, long count, String sha256) {}

    static List<Part> split(Start plan, Path directory, int clients, java.util.function.BooleanSupplier cancelled) throws IOException {
        Files.createDirectory(directory);
        var outputs = new ArrayList<OutputStream>();
        var digests = new ArrayList<MessageDigest>();
        long count = 0;
        Throwable primary = null;
        try {
            for (int i = 0; i < clients; i++) {
                var digest = digest(); digests.add(digest);
                outputs.add(new BufferedOutputStream(new DigestOutputStream(Files.newOutputStream(
                    directory.resolve(i + ".part"), StandardOpenOption.CREATE_NEW), digest)));
            }
            if (plan.requests() != null) {
                for (Input input : plan.requests()) count = write(input, count, plan.qps(), outputs, cancelled);
            } else {
                Path source = Path.of(plan.requests_file());
                var before = Files.readAttributes(source, BasicFileAttributes.class);
                try (var reader = new Reader(source)) {
                    Input input;
                    while ((input = reader.next()) != null) count = write(input, count, plan.qps(), outputs, cancelled);
                }
                var after = Files.readAttributes(source, BasicFileAttributes.class);
                if (before.size() != after.size() || !before.lastModifiedTime().equals(after.lastModifiedTime())
                        || !Objects.equals(before.fileKey(), after.fileKey()))
                    throw new IOException("Input file changed during preparation: " + source);
            }
            if (count < clients) throw new IOException("Need at least one request per Client");
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
            checkInterrupted();
            Path path = directory.resolve(i + ".jsonl").toAbsolutePath();
            Files.move(directory.resolve(i + ".part"), path, StandardCopyOption.ATOMIC_MOVE);
            parts.add(new Part(path.toString(), count / clients + (i < count % clients ? 1 : 0),
                HexFormat.of().formatHex(digests.get(i).digest())));
        }
        return List.copyOf(parts);
    }
    private static long write(Input input, long index, double qps, List<OutputStream> outputs, java.util.function.BooleanSupplier cancelled) throws IOException {
        checkInterrupted();
        if (cancelled.getAsBoolean()) throw new InterruptedIOException("Request preparation stopped");
        long next = Math.addExact(index, 1);
        if (next / qps > 86400) throw new IOException("Playback exceeds 24 hours");
        byte[] bytes = JSON.writeValueAsBytes(input);
        if (bytes.length > MAX_RECORD_BYTES) throw new IOException("Request record exceeds 1 MiB");
        var output = outputs.get((int) (index % outputs.size()));
        output.write(bytes); output.write('\n');
        return next;
    }
    static void checkInterrupted() throws InterruptedIOException {
        if (Thread.currentThread().isInterrupted()) throw new InterruptedIOException("Request preparation/playback interrupted");
    }
    private static MessageDigest digest() {
        try { return MessageDigest.getInstance("SHA-256"); }
        catch (NoSuchAlgorithmException error) { throw new AssertionError(error); }
    }

    static final class Reader implements AutoCloseable {
        private final Path path;
        private final MessageDigest digest = digest();
        private final InputStream stream;
        private long line;
        Reader(Path path) throws IOException {
            this.path = path;
            if (!Files.isRegularFile(path)) throw new IOException("Not a regular request file: " + path);
            stream = new BufferedInputStream(new DigestInputStream(Files.newInputStream(path), digest));
        }
        Input next() throws IOException {
            checkInterrupted();
            line++;
            var bytes = new ByteArrayOutputStream();
            int value;
            while ((value = stream.read()) != -1 && value != '\n') {
                if (bytes.size() == MAX_RECORD_BYTES) throw error("Record exceeds 1 MiB", null);
                bytes.write(value);
            }
            if (value == -1 && bytes.size() == 0) return null;
            try {
                JsonNode node = JSON.readTree(bytes.toByteArray());
                if (node == null || !node.isObject()) throw new IllegalArgumentException("Expected a request object");
                var keys = new HashSet<String>(); node.fieldNames().forEachRemaining(keys::add);
                if (!Set.of("token_ids", "max_new_tokens", "priority").containsAll(keys)
                        || !keys.containsAll(Set.of("token_ids", "max_new_tokens")) || !node.get("token_ids").isArray())
                    throw new IllegalArgumentException("Expected token_ids, max_new_tokens and optional priority only");
                var tokens = new ArrayList<Integer>();
                for (JsonNode token : node.get("token_ids")) tokens.add(integer(token));
                return new Input(tokens, integer(node.get("max_new_tokens")), node.has("priority") ? integer(node.get("priority")) : 50);
            } catch (IllegalArgumentException | com.fasterxml.jackson.core.JsonProcessingException failure) {
                throw error("Invalid text request: " + failure.getMessage(), failure);
            }
        }
        private IOException error(String message, Exception cause) {
            return new IOException(path + ": record " + Math.max(1, line) + ": " + message, cause);
        }
        private static int integer(JsonNode value) {
            if (!value.isIntegralNumber() || !value.canConvertToInt()) throw new IllegalArgumentException("Expected int32");
            return value.intValue();
        }
        void verify(String expected) throws IOException {
            if (!HexFormat.of().formatHex(digest.digest()).equals(expected)) throw error("Shard checksum mismatch", null);
        }
        @Override public void close() throws IOException { stream.close(); }
    }
}
