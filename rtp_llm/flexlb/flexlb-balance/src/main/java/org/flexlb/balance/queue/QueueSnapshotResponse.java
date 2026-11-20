package org.flexlb.balance.queue;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class QueueSnapshotResponse {

    private String filePath;
    private long timestamp;
    private int count;
}
