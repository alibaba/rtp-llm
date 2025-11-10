package org.flexlb.util;

import org.flexlb.constant.CommonConstants;

import java.util.concurrent.ThreadLocalRandom;

public class IdUtils {

    private static final char[] HEX_ARRAY = "0123456789abcdef".toCharArray();

    /**
     * 生成高性能UUID
     * 性能比UUID.randomUUID()高
     */
    public static String fastUuid() {
        ThreadLocalRandom random = ThreadLocalRandom.current();
        long mostSigBits = random.nextLong();
        long leastSigBits = random.nextLong();

        // 手动格式化，避免String.format的开销
        return formatUuid(mostSigBits, leastSigBits);
    }

    public static String getServiceIdByModelName(String modelName) {
        return CommonConstants.FUNCTION + "." + modelName;
    }

    public static String getModelNameByServiceId(String physicalServiceId) {
        return physicalServiceId.substring(CommonConstants.FUNCTION.length() + 1);
    }

    private static String formatUuid(long mostSigBits, long leastSigBits) {
        char[] uuid = new char[32];
        formatUuidPart(mostSigBits, uuid, 0);
        formatUuidPart(leastSigBits, uuid, 16);
        return new String(uuid);
    }

    private static void formatUuidPart(long value, char[] dest, int offset) {
        for (int i = 15; i >= 0; i--) {
            dest[offset + i] = HEX_ARRAY[(int) (value & 0xf)];
            value >>>= 4;
        }
    }
}