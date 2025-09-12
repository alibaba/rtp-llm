package org.flexlb.util;

import org.flexlb.constant.CommonConstants;

import java.util.concurrent.ThreadLocalRandom;
import java.util.zip.CRC32;

public class IdUtils {

    private static final String DIAMOND_PREFIX = "w.";

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

    /**
     * 将字符串固定映射为0-59的数字
     * @param input 输入字符
     * @return 0-59
     */
    public static int stringToFixedNumber(String input) {
        CRC32 crc32 = new CRC32();
        crc32.update(input.getBytes());
        long hash = crc32.getValue();
        // 为了避免负数，取绝对值
        // 对60取模，将32位整数映射到0到59之间
        return (int) Math.abs(hash % 60);
    }

    public static String getServiceIdByModelName(String modelName) {
        return CommonConstants.FUNCTION + "." + modelName;
    }

    public static String getModelNameByServiceId(String physicalServiceId) {
        return physicalServiceId.substring(CommonConstants.FUNCTION.length() + 1);
    }

    public static String getModelNameByDataId(String diamondDataId) {
        // 移除前缀的PREFIX
        return diamondDataId.substring(DIAMOND_PREFIX.length());
    }

    public static String getDiamondDataIdByModelName(String modelName) {
        return DIAMOND_PREFIX + modelName;
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