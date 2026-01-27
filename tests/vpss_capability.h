/**
 * vpss_capability.h - VPSS hardware constraint validation (TEST INFRASTRUCTURE)
 *
 * This file belongs in tests/ directory and is NOT production code.
 * Used to verify VPSS hardware constraints are satisfied in test cases.
 *
 * Production code (VpssMemProcessor) handles VPSS configuration directly.
 * This validator is only for test verification to catch constraint violations early.
 */

#pragma once

#include <string>
#include <vector>
#include "cv/preprocess_config.h"
#include "cv/preprocess_op.h"

/**
 * VPSS hardware constraint validator for test cases.
 * NOT part of production API - testing infrastructure only.
 */
class VpssCapability {
public:
    /**
     * Validate high-level preprocessing configuration against VPSS constraints.
     * @param config Configuration to validate
     * @param error Optional error message output
     * @return true if configuration is valid for VPSS, false otherwise
     */
    static bool validate_config(const lua_cv::PreprocessConfig& config,
                                std::string* error = nullptr);

    /**
     * Validate low-level operation sequence against VPSS constraints.
     * @param ops Operation sequence to validate
     * @param error Optional error message output
     * @return true if operation sequence is valid for VPSS, false otherwise
     */
    static bool validate_ops(const std::vector<lua_cv::PreprocessOp>& ops,
                             std::string* error = nullptr);

    /**
     * Convert high-level config to low-level operation sequence.
     * This is for test validation only. Production code in VpssMemProcessor
     * handles configuration directly without this intermediate step.
     * @param config Configuration to convert
     * @return Operation sequence equivalent to config
     */
    static std::vector<lua_cv::PreprocessOp> config_to_ops(
        const lua_cv::PreprocessConfig& config);

private:
    static bool check_crop_before_resize(const std::vector<lua_cv::PreprocessOp>& ops);
    static bool check_single_resize(const std::vector<lua_cv::PreprocessOp>& ops);
    static bool check_color_conversion(PIXEL_FORMAT_E format);
    static bool check_dimension_limits(uint32_t width, uint32_t height);
};
