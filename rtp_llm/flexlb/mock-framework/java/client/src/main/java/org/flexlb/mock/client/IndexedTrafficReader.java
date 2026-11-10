package org.flexlb.mock.client;

import com.github.luben.zstd.Zstd;
import com.github.luben.zstd.ZstdException;
import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.zip.CRC32;
import org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB;
import org.flexlb.engine.grpc.EngineRpcService.MMPreprocessConfigPB;
import org.flexlb.engine.grpc.EngineRpcService.MultimodalInputPB;

/**
 * Reads requests by ordinal from an immutable traffic dataset and reconstructs their input.
 * File offsets are absolute except request and prefix-page offsets relative to the payload.
 */
final class IndexedTrafficReader implements AutoCloseable {
    private static final byte[] MAGIC = {'M', 'C', 'K', 'I', 'N', 'D', 'E', 'X'};

    private static final int HEADER = 112, INDEX = 16, CHAIN = 20, CHAIN_PAGE = 28, MEDIA = 12, CAP = 8;
    private static final int CHAINS_PER_PAGE = 256, CACHE_CHAIN_PAGES = 128;
    private static final int REQUEST = 26, IMAGE = 16, MAX_IMAGES = 256;
    private static final int MAX_REQUEST = 1024 * 1024;
    private static final int PAGE_BYTES = 4096, CACHE_PAGES = 256;

    final long count, lastArrival, missingOutput;

    final String identity;
    private final FileChannel file;

    private final long fileSize, chains, chainPageCount, media, mediaCount, caps, capCount, payload;

    private final LinkedHashMap<Long, byte[]> pages = new LinkedHashMap<>(16, 0.75f, true);
    private final LinkedHashMap<Long, byte[]> chainPages = new LinkedHashMap<>(16, 0.75f, true);

    IndexedTrafficReader(Path path) throws IOException {
        file = FileChannel.open(path, StandardOpenOption.READ);
        try {
            fileSize = file.size();
            var header = read(0, HEADER);
            var actual = new byte[MAGIC.length];
            header.get(actual);
            if (!Arrays.equals(actual, MAGIC)) throw invalid("unsupported header");
            count = header.getLong();
            chains = header.getLong();
            chainPageCount = header.getLong();
            media = header.getLong();
            mediaCount = header.getLong();
            capCount = header.getLong();
            payload = header.getLong();
            lastArrival = header.getLong();
            missingOutput = header.getLong();
            byte[] digest = new byte[32];
            header.get(digest);
            identity = HexFormat.of().formatHex(digest);
            caps = Math.addExact(media, Math.multiplyExact(mediaCount, MEDIA));
            if (capCount < 0 || capCount > 65535 || missingOutput < 0 || missingOutput > count || count < 1 || chainPageCount < 0 || chainPageCount > Integer.MAX_VALUE / CHAINS_PER_PAGE || mediaCount < 0
                    || lastArrival < 0 || chains != Math.addExact(HEADER, Math.multiplyExact(count, INDEX))
                    || media != Math.addExact(chains, Math.multiplyExact(chainPageCount, CHAIN_PAGE))
                    || payload != Math.addExact(caps, capCount * CAP) || payload >= fileSize)
                throw invalid("invalid table bounds");
        } catch (IOException | RuntimeException error) {
            file.close();
            throw error;
        }
    }

    @Override public void close() throws IOException { file.close(); }

    Entry entry(long index, int imageToken) throws IOException {
        var row = readRequestRow(index);
        long arrival = row.getLong(), leaf = row.getLong();
        int observed = row.getInt(), priority = row.getInt();
        int capId = Short.toUnsignedInt(row.getShort());
        if (capId > capCount) throw invalid("unknown output cap");
        int maximum = capId == 0 ? 0 : checked(table(caps + (capId - 1L) * CAP, CAP)).getInt();
        int images = (row.limit() - REQUEST) / IMAGE;
        if (arrival < 0 || arrival > lastArrival || leaf < 0 || observed < -1 || priority < 0
                || (capId != 0 && (maximum < 1 || observed > maximum)))
            throw invalid("invalid request metadata");

        int[] text = restoreText(leaf, images, imageToken);
        var decoded = buildInput(row, images, text, imageToken);
        return new Entry(arrival, priority, observed, maximum, decoded.input(), decoded.expectedImageTokens());
    }

    private ByteBuffer readRequestRow(long index) throws IOException {
        if (index < 0 || index >= count) throw invalid("request index out of bounds");
        var slot = checked(table(HEADER + index * INDEX, INDEX));
        long offset = slot.getLong();
        int size = slot.getInt();
        if (offset < 0 || offset > fileSize - payload || size < REQUEST + 4
                || size > REQUEST + MAX_IMAGES * IMAGE + 4 || (size - REQUEST - 4) % IMAGE != 0)
            throw invalid("invalid request bounds");
        return checked(table(payload + offset, size));
    }

    private static IOException invalid(String reason) { return new IOException("Indexed traffic: " + reason); }

    private static ByteBuffer checked(ByteBuffer row) throws IOException {
        int end = row.limit() - 4;
        var crc = new CRC32();
        crc.update(row.duplicate().limit(end));
        if ((int) crc.getValue() != row.getInt(end)) throw invalid("record checksum mismatch");
        return row.limit(end);
    }

    private ByteBuffer table(long offset, int size) throws IOException {
        if (offset < HEADER || offset > fileSize - size) throw invalid("record outside file");
        long page = offset / PAGE_BYTES * PAGE_BYTES;
        int relative = (int) (offset - page);
        if (relative + size > PAGE_BYTES) return read(offset, size);
        var bytes = pages.get(page);
        if (bytes == null) {
            bytes = read(page, (int) Math.min(PAGE_BYTES, fileSize - page)).array();
            pages.put(page, bytes);
            if (pages.size() > CACHE_PAGES) pages.pollFirstEntry();
        }
        return ByteBuffer.wrap(bytes, relative, size).slice();
    }

    private ByteBuffer read(long offset, int size) throws IOException {
        var bytes = ByteBuffer.allocate(size);
        while (bytes.hasRemaining()) {
            if (file.read(bytes, offset + bytes.position()) < 0) throw new EOFException("Truncated indexed traffic");
        }
        return bytes.flip();
    }

    private int[] restoreText(long leaf, int images, int imageToken) throws IOException {
        var reversed = new ArrayList<Chain>();
        int textCount = 0;
        for (long node = leaf; node != 0;) {
            var chain = chain(node);
            int length = Math.toIntExact(node - chain.first + 1);
            textCount = Math.addExact(textCount, length);
            if (textCount > MAX_REQUEST - images) throw invalid("request exceeds token limit");
            reversed.add(new Chain(chain.first, chain.parent, length, chain.symbol));
            node = chain.parent;
        }

        int[] text = new int[textCount];
        for (int i = reversed.size() - 1, position = 0; i >= 0; i--) {
            var chain = reversed.get(i);
            text[position] = token(chain.symbol, imageToken);
            if (imageToken == 0) Arrays.fill(text, position + 1, position + chain.size, 1);
            position += chain.size;
        }
        return text;
    }

    private Chain chain(long node) throws IOException {
        long lo = 0, hi = chainPageCount;
        while (lo < hi) {
            long mid = lo + (hi - lo) / 2;
            long first = checked(table(chains + mid * CHAIN_PAGE, CHAIN_PAGE)).getLong();
            if (node < first) hi = mid;
            else lo = mid + 1;
        }
        if (lo == 0) throw invalid("unknown prefix node");
        long page = lo - 1;
        var rows = chainPage(page);
        int left = 0, right = rows.capacity() / CHAIN;
        while (left < right) {
            int index = left + (right - left) / 2;
            int offset = index * CHAIN;
            long first = rows.getLong(offset), parent = rows.getLong(offset + 8);
            int size = rows.getInt(offset + 16);
            if (first < 1 || parent < 0 || parent >= first || size < 1) throw invalid("invalid chain");
            if (node < first) right = index;
            else if (node - first >= size) left = index + 1;
            else return new Chain(first, parent, size, (int) (page * CHAINS_PER_PAGE + index + 1));
        }
        throw invalid("unknown prefix node");
    }

    private ByteBuffer chainPage(long index) throws IOException {
        var cached = chainPages.get(index);
        if (cached == null) {
            var pointer = checked(table(chains + index * CHAIN_PAGE, CHAIN_PAGE));
            long first = pointer.getLong(), offset = pointer.getLong();
            int size = pointer.getInt(), count = pointer.getInt();
            if (first < 1 || count < 1 || count > CHAINS_PER_PAGE || size < 1 || size > 8192
                    || offset < 0 || offset > fileSize - payload - size)
                throw invalid("invalid chain page bounds");
            try {
                cached = Zstd.decompress(read(payload + offset, size).array(), count * CHAIN);
            } catch (ZstdException error) {
                throw new IOException("Indexed traffic: invalid compressed chain page", error);
            }
            if (cached.length != count * CHAIN || ByteBuffer.wrap(cached).getLong() != first)
                throw invalid("invalid chain page contents");
            chainPages.put(index, cached);
            if (chainPages.size() > CACHE_CHAIN_PAGES) chainPages.pollFirstEntry();
        }
        return ByteBuffer.wrap(cached);
    }

    private static int token(int symbol, int imageToken) {
        return imageToken >= 0 && symbol >= imageToken ? symbol + 1 : symbol;
    }

    private DecodedInput buildInput(ByteBuffer row, int images, int[] text, int imageToken) throws IOException {
        int textCount = text.length;
        var request = GenerateInputPB.newBuilder();
        var expected = new ArrayList<Integer>(images);
        int previous = 0;
        for (int i = 0; i < images; i++) {
            long id = row.getLong();
            int at = row.getInt(), expectedTokens = row.getInt();
            if (id < 1 || id > mediaCount || at < previous || at > textCount || expectedTokens < 0)
                throw invalid("invalid image reference");
            var geometry = checked(table(media + (id - 1) * MEDIA, MEDIA));
            int width = geometry.getInt(), height = geometry.getInt();
            if (width < 1 || width > 32768 || height < 1 || height > 32768) throw invalid("invalid media dimensions");
            for (int t = previous; t < at; t++) request.addTokenIds(text[t]);
            request.addMultimodalInputs(MultimodalInputPB.newBuilder().setMultimodalType(1).setMultimodalUrl("mock-image:" + id)
                .setMmPreprocessConfig(MMPreprocessConfigPB.newBuilder().setWidth(width).setHeight(height)));
            request.addTokenIds(imageToken);
            previous = at;
            expected.add(expectedTokens);
        }
        for (int t = previous; t < textCount; t++) request.addTokenIds(text[t]);
        var input = request.build();
        if (input.getTokenIdsCount() == 0 || input.getSerializedSize() > MAX_REQUEST)
            throw invalid("invalid or oversized input");
        return new DecodedInput(input, List.copyOf(expected));
    }

    record Entry(long arrivalMs, int priority, int observedOutputTokens, int maxOutputTokens,
                 GenerateInputPB input, List<Integer> expectedImageTokens) {}

    private record DecodedInput(GenerateInputPB input, List<Integer> expectedImageTokens) {}

    private record Chain(long first, long parent, int size, int symbol) {}
}
