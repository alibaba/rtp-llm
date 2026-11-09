package org.flexlb.mock.client;

import com.github.luben.zstd.ZstdCompressCtx;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.zip.CRC32;
import org.flexlb.mock.client.ClientProtocol.Start;

final class InputFixture {
    static Path write(Path path, List<Long> arrivals) throws Exception {
        var rows = new ArrayList<ByteBuffer>();
        for (long arrival : arrivals) rows.add(ByteBuffer.allocate(26).putLong(arrival).putLong(3).putInt(-1).putInt(20).putShort((short) 1));
        return indexed(path, rows, ByteBuffer.allocate(20).putLong(1).putLong(0).putInt(3), arrivals.getLast(), List.of(), List.of(3), 0);
    }

    static Path indexed(Path path, List<ByteBuffer> rows, ByteBuffer chains, long last, List<int[]> media, List<Integer> caps, long missingOutput) throws Exception {
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
        for (int cap : caps) record(tables, ByteBuffer.allocate(4).putInt(cap));
        payload.writeTo(tables);
        tables.write(page);
        long chainIndex = 112L + rows.size() * 16L;
        long mediaOffset = chainIndex + 28;
        try (var out = new DataOutputStream(Files.newOutputStream(path, StandardOpenOption.CREATE_NEW))) {
            out.writeBytes("MCKINDEX");
            for (long value : new long[]{rows.size(), chainIndex, 1, mediaOffset, media.size(), caps.size(),
                    mediaOffset + media.size() * 12L + caps.size() * 8L, last, missingOutput})
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
        var path = write(directory.resolve(UUID.randomUUID() + ".mckidx"), Collections.nCopies(count, 0L));
        return new Start(masters, path.toString(), qps, null, slots, queue, timeout, drain);
    }
}
