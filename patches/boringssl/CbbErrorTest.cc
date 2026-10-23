#include <openssl/bytestring.h>
#include <gtest/gtest.h>
#include <array>

TEST(CbbErrorTest, FailedAsn1WritesClearTheExpiredChild) {
    for (size_t capacity = 0; capacity < 3; ++capacity) {
        for (bool octet_string : {false, true}) {
            std::array<uint8_t, 3> storage{};
            CBB                    cbb;
            ASSERT_TRUE(CBB_init_fixed(&cbb, storage.data(), capacity));
            const uint8_t value = 7;
            const int result = octet_string ? CBB_add_asn1_octet_string(&cbb, &value, 1) : CBB_add_asn1_bool(&cbb, 1);
            EXPECT_EQ(result, 0);
            EXPECT_EQ(cbb.child, nullptr);
            EXPECT_EQ(CBB_flush(&cbb), 0);
            CBB_cleanup(&cbb);
        }
    }
}

TEST(CbbErrorTest, SuccessfulAsn1EncodingIsUnchanged) {
    for (bool octet_string : {false, true}) {
        std::array<uint8_t, 3> storage{};
        CBB                    cbb;
        ASSERT_TRUE(CBB_init_fixed(&cbb, storage.data(), storage.size()));
        const uint8_t value = 7;
        ASSERT_TRUE(octet_string ? CBB_add_asn1_octet_string(&cbb, &value, 1) : CBB_add_asn1_bool(&cbb, 1));
        EXPECT_EQ(cbb.child, nullptr);
        EXPECT_EQ(storage[0], octet_string ? 4 : 1);
        EXPECT_EQ(storage[1], 1);
        EXPECT_EQ(storage[2], octet_string ? 7 : 255);
        CBB_cleanup(&cbb);
    }
}
