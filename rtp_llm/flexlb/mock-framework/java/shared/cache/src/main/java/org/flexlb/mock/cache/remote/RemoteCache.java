package org.flexlb.mock.cache.remote;

import java.util.List;

/** Logical remote KV operations. KvCache supplies asynchronous execution and owns local blocks.
 * Implementations may block, must honor the monotonic deadline, and must stop accessing a
 * transfer's resources before returning or throwing. Interruption is a cancellation request,
 * not evidence that I/O has stopped. No concrete storage or payload SDK is installed here.
 */
public interface RemoteCache extends AutoCloseable {
    /** Backend-owned read credentials. The count is the total continuous prefix, including local blocks. */
    interface Match { int prefixBlocks(); }

    Match match(List<Long> keys, int localBlocks, long deadlineNanos) throws Exception;
    /** Read the missing suffix. Successful return means every requested logical block is ready. */
    void read(Match match, List<Long> missingKeys, long deadlineNanos) throws Exception;
    /** Return the number of successfully published blocks, after backend confirmation (0 if already present). */
    int write(List<Long> keys, long deadlineNanos) throws Exception;
    @Override void close();
}
