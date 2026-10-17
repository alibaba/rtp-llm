package org.flexlb.util;

import org.flexlb.constant.CommonConstants;

import java.util.HexFormat;
import java.util.concurrent.ThreadLocalRandom;

public class IdUtils {

    /**
     * Generate high-performance UUID
     * Higher performance than UUID.randomUUID()
     */
    public static String fastUuid() {
        ThreadLocalRandom random = ThreadLocalRandom.current();
        long mostSigBits = random.nextLong();
        long leastSigBits = random.nextLong();

        return HexFormat.of().toHexDigits(mostSigBits)
                + HexFormat.of().toHexDigits(leastSigBits);
    }

    public static String getModelNameByServiceId(String physicalServiceId) {
        return physicalServiceId.substring(CommonConstants.FUNCTION.length() + 1);
    }

}
