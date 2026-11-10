package org.flexlb.mock.engine.vit;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.GeneralSecurityException;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPOutputStream;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

/** Temporary, opt-in anonymous media geometry for a Frontend capture window. */
final class VitGeometryCapture {
    private static final int MAX_IMAGES = 64 * 1024;
    private static final long DRAIN_MS = 60_000;
    private final long start;
    private final long end;
    private final Mac mac;
    private final ArrayBlockingQueue<String> pending = new ArrayBlockingQueue<>(2048);
    private final Set<String> seen = new HashSet<>();
    private final Path directory;
    private int dropped;
    private boolean closed;

    static VitGeometryCapture fromEnvironment() {
        String value = System.getenv("MOCK_CAPTURE_START_MS");
        if (value == null) return null;
        long start = Long.parseLong(value);
        long end = Long.parseLong(System.getenv("MOCK_CAPTURE_END_MS"));
        byte[] key = HexFormat.of().parseHex(System.getenv("MOCK_CAPTURE_KEY"));
        if (key.length != 32 || end <= start || end - start > 300_000) {
            throw new IllegalArgumentException("Invalid ViT capture window");
        }
        return end + DRAIN_MS > System.currentTimeMillis() ? new VitGeometryCapture(start, end, key) : null;
    }

    private VitGeometryCapture(long start, long end, byte[] key) {
        this.start = start;
        this.end = end;
        directory = Path.of("/tmp/rtp-vit-capture", Long.toString(start));
        try {
            mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(key, "HmacSHA256"));
        } catch (GeneralSecurityException failure) {
            throw new IllegalStateException("Cannot initialize ViT geometry capture", failure);
        }
        Thread.ofPlatform().name("vit-geometry-capture").daemon().start(this::write);
    }

    synchronized void record(VitGeometry.Image image) {
        long now = System.currentTimeMillis();
        if (closed || now < start || now >= end + DRAIN_MS) return;
        String identity = digest("image", image.key());
        if (seen.contains(identity)) return;
        if (seen.size() >= MAX_IMAGES || !pending.offer("[\"image\",\"" + identity + "\","
                + image.width() + "," + image.height() + "]")) {
            dropped++;
            return;
        }
        seen.add(identity);
    }

    private String digest(String namespace, String value) {
        mac.update(namespace.getBytes(StandardCharsets.UTF_8));
        return HexFormat.of().formatHex(mac.doFinal(value.getBytes(StandardCharsets.UTF_8)), 0, 16);
    }

    private void write() {
        String shard = UUID.randomUUID().toString().replace("-", "");
        Path target = directory.resolve("vit-" + shard + ".jsonl.gz");
        Path partial = directory.resolve(target.getFileName() + ".part");
        try {
            Files.createDirectories(directory, PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rwx------")));
            Files.createFile(partial, PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rw-------")));
            String check;
            synchronized (this) {
                check = digest("check", "");
            }
            try (var stream = new BufferedWriter(new OutputStreamWriter(
                    new GZIPOutputStream(Files.newOutputStream(partial)), StandardCharsets.UTF_8))) {
                emit(stream, "[\"geometry\",3," + start + "," + end + ",\"" + shard + "\",\"" + check + "\"]");
                while (true) {
                    String row = pending.poll(100, TimeUnit.MILLISECONDS);
                    if (row != null) {
                        emit(stream, row);
                    } else {
                        synchronized (this) {
                            if (System.currentTimeMillis() >= end + DRAIN_MS && pending.isEmpty()) {
                                closed = true;
                                emit(stream, "[\"end\"," + seen.size() + "," + dropped + "]");
                                break;
                            }
                        }
                    }
                }
            }
            Files.move(partial, target, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException | InterruptedException error) {
            synchronized (this) {
                closed = true;
            }
            System.getLogger(VitGeometryCapture.class.getName()).log(System.Logger.Level.ERROR,
                    "ViT geometry capture failed: " + error.getClass().getSimpleName());
        }
    }

    private static void emit(BufferedWriter stream, String row) throws IOException {
        stream.write(row);
        stream.newLine();
    }
}
