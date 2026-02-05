#include <gtest/gtest.h>
#include "node/error_codes.h"

using namespace node;

TEST(ErrorCodes, Values) {
    // Verify error code values match sscma-node protocol
    EXPECT_EQ(MA_OK, 0);
    EXPECT_EQ(MA_ENOENT, 2);
    EXPECT_EQ(MA_EAGAIN, 11);
    EXPECT_EQ(MA_ENOMEM, 12);
    EXPECT_EQ(MA_EBUSY, 16);
    EXPECT_EQ(MA_EEXIST, 17);
    EXPECT_EQ(MA_EINVAL, 22);
}

TEST(ErrorCodes, SuccessCheck) {
    // MA_OK should be 0 (success)
    EXPECT_EQ(MA_OK, 0);
    // All other codes are errors
    EXPECT_NE(MA_EAGAIN, 0);
    EXPECT_NE(MA_ENOMEM, 0);
    EXPECT_NE(MA_EINVAL, 0);
}
