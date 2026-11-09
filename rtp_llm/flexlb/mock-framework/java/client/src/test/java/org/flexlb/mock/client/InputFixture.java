package org.flexlb.mock.client;

import com.github.luben.zstd.ZstdCompressCtx;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.zip.CRC32;
import org.flexlb.mock.client.input.ClientInput.Request;
import org.flexlb.mock.client.ClientProtocol.Start;

final class InputFixture {
    static Request request() { return Request.newBuilder().addAllTokenIds(List.of(1, 2, 3)).setMaxNewTokens(3).setPriority(20).build(); }

    static Path write(Path path, List<Request> requests) throws Exception {
        var rows = new ArrayList<ByteBuffer>();
        long last = 0;
        for (var request : requests) {
            last = request.getArrivalOffsetMs();
            rows.add(ByteBuffer.allocate(24).putLong(last).putLong(3).putInt(request.getMaxNewTokens()).putInt(request.getPriority()));
        }
        return indexed(path, rows, ByteBuffer.allocate(20).putLong(1).putLong(0).putInt(3), last, List.of());
    }

    static Path indexed(Path path, List<ByteBuffer> rows, ByteBuffer chains, long last, List<int[]> media) throws Exception {
        var payload = new ByteArrayOutputStream();
        var tables = new ByteArrayOutputStream();
        for (var row : rows) {
            record(tables, ByteBuffer.allocate(12).putLong(payload.size()).putInt(row.capacity() + 4));
            record(payload, row);
        }
        byte[] page;
        try (var compressor = new ZstdCompressCtx()) {
            page = compressor.setChecksum(true).compress(chains.array());
        }
        record(tables, ByteBuffer.allocate(24).putLong(1).putLong(payload.size()).putInt(page.length).putInt(chains.capacity() / 20));
        for (var image : media) record(tables, ByteBuffer.allocate(8).putInt(image[0]).putInt(image[1]));
        payload.writeTo(tables);
        tables.write(page);
        long chainIndex = 96L + rows.size() * 16L;
        long mediaOffset = chainIndex + 28;
        try (var out = new DataOutputStream(Files.newOutputStream(path, StandardOpenOption.CREATE_NEW))) {
            out.writeBytes("MCKIDX01");
            for (long value : new long[]{rows.size(), chainIndex, 1, mediaOffset, media.size(), mediaOffset + media.size() * 12L, last})
                out.writeLong(value);
            out.write(MessageDigest.getInstance("SHA-256").digest(tables.toByteArray()));
            tables.writeTo(out);
        }
        return path;
    }

    private static void record(ByteArrayOutputStream out, ByteBuffer row) throws IOException {
        out.write(row.array());
        var checksum = new CRC32();
        checksum.update(row.array());
        new DataOutputStream(out).writeInt((int) checksum.getValue());
    }

    static Start plan(Path directory, List<String> masters, int count, double qps, int slots, int queue, long timeout, long drain) throws Exception {
        var path = write(directory.resolve(UUID.randomUUID() + ".mckidx"), Collections.nCopies(count, request()));
        return new Start(masters, path.toString(), qps, null, slots, queue, timeout, drain);
    }
}
