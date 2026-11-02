package org.flexlb.mock.client;

import java.io.*;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.security.*;
import java.util.*;
import org.flexlb.mock.client.ClientProtocol.*;

/** Validate the complete compressed source before publishing immutable compressed shards. */
final class RequestFiles {
    record Part(String path, long count, String sha256) {}
    static List<Part> split(Start plan, Path directory, int clients, java.util.function.BooleanSupplier cancelled) throws IOException {
        Files.createDirectory(directory);
        var outputs = new ArrayList<InputFile.Writer>();
        var digests = new ArrayList<MessageDigest>();
        long count = 0, previousArrival = -1;
        Throwable primary = null;
        try {
            for (int i = 0; i < clients; i++) {
                var digest = InputFile.digest(); digests.add(digest);
                outputs.add(new InputFile.Writer(new BufferedOutputStream(new DigestOutputStream(Files.newOutputStream(
                    directory.resolve(i + ".part"), StandardOpenOption.CREATE_NEW), digest))));
            }
            Path source = Path.of(plan.requests_file());
            var before = Files.readAttributes(source, BasicFileAttributes.class);
            try (var reader = new InputFile.Reader(source)) {
                org.flexlb.mock.client.input.ClientInput.Request input;
                while ((input = reader.next()) != null) {
                    InputFile.checkInterrupted();
                    if (cancelled.getAsBoolean()) throw new InterruptedIOException("Request preparation stopped");
                    long next = Math.addExact(count, 1);
                    if (plan.qps() != null && next / plan.qps() > 86400) throw new IOException("Playback exceeds 24 hours");
                    Player.delayNanos(count, input, plan.qps(), plan.replay_speed());
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
            InputFile.checkInterrupted();
            Path path = directory.resolve(i + ".binpb.zst").toAbsolutePath();
            Files.move(directory.resolve(i + ".part"), path, StandardCopyOption.ATOMIC_MOVE);
            parts.add(new Part(path.toString(), count / clients + (i < count % clients ? 1 : 0), HexFormat.of().formatHex(digests.get(i).digest())));
        }
        return List.copyOf(parts);
    }
}
