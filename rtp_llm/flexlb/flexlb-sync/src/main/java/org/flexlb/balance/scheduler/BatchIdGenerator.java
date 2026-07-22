package org.flexlb.balance.scheduler;

import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Generates a Snowflake-compatible batch ID.
 *
 * <p>Structure (64 bits):
 * <pre>
 *  31 bits: relative timestamp (seconds since 2020-01-01) — supports ~68 years
 *  12 bits: master instance id (hash of address and process identity)
 *  21 bits: sequence number — 2,097,152 batches per second per master
 * </pre>
 *
 * <p>Uniqueness assumes the wall clock does not move backwards and one master
 * generates fewer than 2,097,152 batches in the same second.
 */
public class BatchIdGenerator {

    private static final long EPOCH_2020_SECONDS = 1577836800L; // 2020-01-01 UTC
    private static final long TIMESTAMP_BITS = 31L;
    private static final long SEQUENCE_BITS = 21L;
    private static final long MASTER_ID_BITS = 12L;
    private static final long SEQUENCE_MASK = (1L << SEQUENCE_BITS) - 1;
    private static final long MASTER_ID_SHIFT = SEQUENCE_BITS;
    private static final long TIMESTAMP_SHIFT = SEQUENCE_BITS + MASTER_ID_BITS;
    private static final AtomicLong SEQUENCE = new AtomicLong();

    private final long masterId;

    public BatchIdGenerator(String localIp, int port) {
        this.masterId = computeMasterId(localIp, port,
                ProcessHandle.current().pid(),
                ManagementFactory.getRuntimeMXBean().getStartTime());
    }

    /** Generates the next batch ID. Thread-safe within one master process. */
    public long nextBatchId() {
        long timestamp = System.currentTimeMillis() / 1000L - EPOCH_2020_SECONDS;
        if (timestamp < 0) {
            throw new IllegalStateException("system clock is before the batch ID epoch");
        }
        if ((timestamp >>> TIMESTAMP_BITS) != 0) {
            throw new IllegalStateException("batch ID timestamp overflow");
        }
        long sequenceId = SEQUENCE.getAndIncrement() & SEQUENCE_MASK;
        return (timestamp << TIMESTAMP_SHIFT)
                | (masterId << MASTER_ID_SHIFT)
                | sequenceId;
    }

    /**
     * Computes a 12-bit master process identifier via SHA256.
     */
    private static long computeMasterId(String localIp,
                                        int port,
                                        long processId,
                                        long processStartTimeMs) {
        try {
            String input = localIp + ":" + port + ":" + processId + ":" + processStartTimeMs;
            byte[] hash = MessageDigest.getInstance("SHA-256")
                    .digest(input.getBytes(StandardCharsets.UTF_8));
            // Take the low 12 bits of the hash
            return (hash[hash.length - 1] & 0xFF) | ((hash[hash.length - 2] & 0x0F) << 8);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-256 not available", e);
        }
    }
}
