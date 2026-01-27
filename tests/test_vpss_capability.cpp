/**
 * test_vpss_capability.cpp - VPSS capability validation tests
 *
 * Tests the VpssCapability validation logic for PreprocessConfig and PreprocessOp.
 * These tests run locally (no device required) and validate constraint checking.
 */

#include <gtest/gtest.h>
#include "vpss_capability.h"

using namespace lua_cv;

TEST(VpssCapability, ValidConfigBasic) {
    PreprocessConfig config;
    config.input_width = 1920;
    config.input_height = 1080;
    config.output_width = 640;
    config.output_height = 640;
    config.output_format = PIXEL_FORMAT_RGB_888_PLANAR;
    config.letterbox = true;

    std::string error;
    EXPECT_TRUE(VpssCapability::validate_config(config, &error)) << error;
}

TEST(VpssCapability, InvalidConfigZeroDimensions) {
    PreprocessConfig config;
    config.input_width = 0;
    config.input_height = 0;
    config.output_width = 640;
    config.output_height = 640;

    std::string error;
    EXPECT_FALSE(VpssCapability::validate_config(config, &error));
    EXPECT_FALSE(error.empty());
}

TEST(VpssCapability, InvalidConfigTooLarge) {
    PreprocessConfig config;
    config.input_width = 8192;
    config.input_height = 8192;
    config.output_width = 640;
    config.output_height = 640;

    std::string error;
    EXPECT_FALSE(VpssCapability::validate_config(config, &error));
    EXPECT_FALSE(error.empty());
}

TEST(VpssCapability, InvalidConfigNormalize) {
    PreprocessConfig config;
    config.input_width = 1920;
    config.input_height = 1080;
    config.output_width = 640;
    config.output_height = 640;
    config.normalized = true;

    std::string error;
    EXPECT_FALSE(VpssCapability::validate_config(config, &error));
    EXPECT_NE(error.find("normalize"), std::string::npos);
}

TEST(VpssCapability, ValidOpsSequence) {
    std::vector<PreprocessOp> ops;
    ops.push_back(PreprocessOp::resize(640, 640, true));
    ops.push_back(PreprocessOp::pad(0, 0, 0, 0, 114));
    ops.push_back(PreprocessOp::color(PIXEL_FORMAT_RGB_888_PLANAR));

    std::string error;
    EXPECT_TRUE(VpssCapability::validate_ops(ops, &error)) << error;
}

TEST(VpssCapability, InvalidOpsCropAfterResize) {
    std::vector<PreprocessOp> ops;
    ops.push_back(PreprocessOp::resize(640, 640, false));
    ops.push_back(PreprocessOp::crop(0, 0, 320, 320));

    std::string error;
    EXPECT_FALSE(VpssCapability::validate_ops(ops, &error));
    EXPECT_NE(error.find("CROP"), std::string::npos);
}

TEST(VpssCapability, InvalidOpsMultipleResize) {
    std::vector<PreprocessOp> ops;
    ops.push_back(PreprocessOp::resize(800, 800, false));
    ops.push_back(PreprocessOp::resize(640, 640, false));

    std::string error;
    EXPECT_FALSE(VpssCapability::validate_ops(ops, &error));
    EXPECT_NE(error.find("one RESIZE"), std::string::npos);
}

TEST(VpssCapability, ConfigToOpsLetterbox) {
    PreprocessConfig config;
    config.input_width = 1920;
    config.input_height = 1080;
    config.output_width = 640;
    config.output_height = 640;
    config.output_format = PIXEL_FORMAT_RGB_888_PLANAR;
    config.letterbox = true;
    config.pad_value = 114;

    auto ops = VpssCapability::config_to_ops(config);

    ASSERT_GE(ops.size(), 2u);
    EXPECT_EQ(ops[0].type, PreprocessOp::RESIZE);
    EXPECT_TRUE(ops[0].params.resize.keep_aspect_ratio);

    bool has_pad = false;
    bool has_color = false;
    for (const auto& op : ops) {
        if (op.type == PreprocessOp::PAD) {
            has_pad = true;
            EXPECT_EQ(op.params.pad.value, 114u);
        } else if (op.type == PreprocessOp::COLOR) {
            has_color = true;
            EXPECT_EQ(op.params.color.format, PIXEL_FORMAT_RGB_888_PLANAR);
        }
    }
    EXPECT_TRUE(has_pad);
    EXPECT_TRUE(has_color);

    std::string error;
    EXPECT_TRUE(VpssCapability::validate_ops(ops, &error)) << error;
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
