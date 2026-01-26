/**
 * test_input_source_camera.cpp - CameraSource tests
 *
 * NOTE: Camera + VPSS processing tests have been consolidated into
 * CameraCaptureTest to avoid multiple camera open/close cycles which
 * are problematic in VI_OFFLINE_VPSS_ONLINE mode.
 *
 * See: test_camera_capture.cpp
 *   - CameraCaptureTest.StreamChannel (Camera→Stream scenario)
 *   - CameraCaptureTest.InferChannel (Camera→Infer scenario)
 *   - CameraCaptureTest.VpssPipelineToTensor (full pipeline)
 */

#include "test_common.h"

TEST(InputSourceCamera, ConsolidatedIntoCameraCaptureTest) {
    GTEST_SKIP() << "Test consolidated into CameraCaptureTest fixture";
}
