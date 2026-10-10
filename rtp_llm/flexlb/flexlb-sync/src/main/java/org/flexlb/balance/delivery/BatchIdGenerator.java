package org.flexlb.balance.delivery;

import org.flexlb.util.Logger;

import java.lang.management.ManagementFactory;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.function.LongSupplier;

/**
 * Generates positive batch IDs with 30 timestamp bits (seconds since 2020),
 * 12 process-instance bits, and 21 sequence bits. The sign bit stays clear.
 * The sequence restarts each second; exhaustion and rollback of the timestamp
 * in seconds fail immediately. Millisecond changes within a second are allowed.
 *
 * <p>The process instance is hashed into only 4,096 values. Uniqueness is
 * guaranteed within one generator, not across processes with colliding hashes.
 * Production should create one generator per process.
 */
public final class BatchIdGenerator {

    private static final int TIMESTAMP_BITS = 30;
    private static final int PROCESS_INSTANCE_BITS = 12;
    private static final int SEQUENCE_BITS = 21;
    private static final int TIMESTAMP_SHIFT = PROCESS_INSTANCE_BITS + SEQUENCE_BITS;
    private static final long EPOCH_SECONDS = 1_577_836_800L;
    private static final long PROCESS_INSTANCE_MASK = (1L << PROCESS_INSTANCE_BITS) - 1L;
    private static final long SEQUENCE_LIMIT = 1L << SEQUENCE_BITS;

    private final long processInstanceId;
    private final LongSupplier currentTimeMillis;
    private long lastTimestampSeconds = -1L;
    private long nextSequence;

    BatchIdGenerator(long processInstanceId, LongSupplier currentTimeMillis) {
        if (processInstanceId < 0L || processInstanceId > PROCESS_INSTANCE_MASK) {
            throw new IllegalArgumentException("processInstanceId must fit in 12 bits");
        }
        this.processInstanceId = processInstanceId;
        this.currentTimeMillis = Objects.requireNonNull(currentTimeMillis, "currentTimeMillis");
    }

    public static BatchIdGenerator forCurrentProcess(int serverPort) {
        String processIdentity = detectLocalIp() + ":" + serverPort + ":"
                + ProcessHandle.current().pid() + ":"
                + ManagementFactory.getRuntimeMXBean().getStartTime();
        return new BatchIdGenerator(hashProcessInstance(processIdentity), System::currentTimeMillis);
    }

    public synchronized long nextId() {
        long timestampSeconds = TimeUnit.MILLISECONDS.toSeconds(currentTimeMillis.getAsLong())
                - EPOCH_SECONDS;
        if (timestampSeconds < 0L || (timestampSeconds >>> TIMESTAMP_BITS) != 0L) {
            throw new IllegalStateException(timestampSeconds < 0L
                    ? "system clock is before the batch ID epoch"
                    : "batch ID timestamp overflow");
        }
        if (timestampSeconds < lastTimestampSeconds) {
            throw new IllegalStateException("system clock moved backwards while generating batch IDs");
        }
        long sequence = timestampSeconds == lastTimestampSeconds ? nextSequence : 0L;
        // Zero denotes no batch in scheduler APIs; the first epoch second must skip it.
        if (timestampSeconds == 0L && processInstanceId == 0L && sequence == 0L) {
            sequence = 1L;
        }
        if (sequence >= SEQUENCE_LIMIT) {
            throw new IllegalStateException("batch ID sequence exhausted for the current second");
        }
        lastTimestampSeconds = timestampSeconds;
        nextSequence = sequence + 1L;
        return (timestampSeconds << TIMESTAMP_SHIFT)
                | (processInstanceId << SEQUENCE_BITS)
                | sequence;
    }

    private static long hashProcessInstance(String processIdentity) {
        try {
            byte[] hash = MessageDigest.getInstance("SHA-256")
                    .digest(processIdentity.getBytes(StandardCharsets.UTF_8));
            long lowBytes = Byte.toUnsignedInt(hash[hash.length - 1])
                    | (Byte.toUnsignedInt(hash[hash.length - 2]) << Byte.SIZE);
            return lowBytes & PROCESS_INSTANCE_MASK;
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 not available", impossible);
        }
    }

    private static String detectLocalIp() {
        try {
            return InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException failure) {
            Logger.warn("Failed to detect local IP for batch IDs; using loopback", failure);
            return "127.0.0.1";
        }
    }
}
