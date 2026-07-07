package org.flexlb.balance.scheduler;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Snowflake-like batch ID generator, aligned with the Python request_id design.
 *
 * <p>Structure (64 bits, same layout as request_id):
 * <pre>
 *  40 bits: relative timestamp (ms since 2020-01-01)  — supports ~35 years
 *  12 bits: master instance id (SHA256 of localIp:port) — 4096 master instances
 *  12 bits: sequence number (per-generator AtomicLong)  — 4096 batches per ms per master
 * </pre>
 *
 * <p>This ensures batch_id is globally unique across master failovers and restarts,
 * consistent with the request_id snowflake pattern in
 * {@code rtp_llm/frontend/request_id_generator.py}.
 */
public class BatchIdGenerator {

    private static final long EPOCH_2020_MS = 1577836800000L; // 2020-01-01 UTC
    private static final long SEQUENCE_BITS = 12L;
    private static final long MASTER_ID_BITS = 12L;
    private static final long SEQUENCE_MASK = (1L << SEQUENCE_BITS) - 1;   // 4095
    private static final long MASTER_ID_MASK = (1L << MASTER_ID_BITS) - 1; // 4095
    private static final long MASTER_ID_SHIFT = SEQUENCE_BITS;              // 12
    private static final long TIMESTAMP_SHIFT = SEQUENCE_BITS + MASTER_ID_BITS; // 24

    private final long masterId;
    private final AtomicLong sequence = new AtomicLong(0);

    public BatchIdGenerator(String localIp, int port) {
        this.masterId = computeMasterId(localIp, port);
    }

    /**
     * Generates the next globally-unique batch ID.
     * Thread-safe via AtomicLong CAS.
     */
    public long nextBatchId() {
        long currentTs = (System.currentTimeMillis() - EPOCH_2020_MS) & 0xFFFFFFFFFFL; // 40-bit mask, aligned with Python request_id_generator.py
        // Sequence is a global counter (not per-ms reset), masked to 12 bits.
        // Safe as long as batch QPS < 4,096,000 per master, well beyond batch scheduling workloads.
        // Aligned with Python request_id_generator.py design.
        long seq = sequence.getAndIncrement() & SEQUENCE_MASK;
        return (currentTs << TIMESTAMP_SHIFT) | (masterId << MASTER_ID_SHIFT) | seq;
    }

    /**
     * Computes a 12-bit master identifier from localIp:port via SHA256,
     * mirroring the request_id machine_id approach.
     */
    private static long computeMasterId(String localIp, int port) {
        try {
            String input = localIp + ":" + port;
            byte[] hash = MessageDigest.getInstance("SHA-256")
                    .digest(input.getBytes(StandardCharsets.UTF_8));
            // Take the low 12 bits of the hash
            return (hash[hash.length - 1] & 0xFF) | ((hash[hash.length - 2] & 0x0F) << 8);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-256 not available", e);
        }
    }
}
