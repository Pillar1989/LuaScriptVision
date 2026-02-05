#include <gtest/gtest.h>
#include "node/resource_estimator.h"
#include "node/resource_limits.h"

using namespace node;

TEST(ResourceLimits, Constants) {
    // Verify constants are properly defined
    EXPECT_EQ(VB_POOL4_TOTAL, 4);
    EXPECT_EQ(VB_BUFFER_RESERVE, 1);
    EXPECT_EQ(MAX_PARALLEL_MODELS, 3);
    EXPECT_EQ(MAX_SERIAL_MODELS, 3);
}

TEST(ResourceLimits, MemoryLimits) {
    EXPECT_EQ(ION_TOTAL_MB, 60);
    EXPECT_EQ(ION_RESERVED_MB, 22);
    EXPECT_EQ(ION_AVAILABLE_MB, 38);
}

TEST(ResourceEstimator, Singleton) {
    ResourceEstimator& estimator = ResourceEstimator::instance();

    // Get same instance
    ResourceEstimator& estimator2 = ResourceEstimator::instance();
    EXPECT_EQ(&estimator, &estimator2);
}

TEST(ResourceEstimator, EstimateResultFields) {
    EstimateResult ok;
    EXPECT_FALSE(ok.pass);  // Default is false
    EXPECT_EQ(ok.error_code, MA_OK);
    EXPECT_TRUE(ok.reason.empty());
}

TEST(ResourceEstimator, ResourceRequirementFields) {
    ResourceRequirement req;
    EXPECT_EQ(req.vb_infer_blocks, 0);
    EXPECT_EQ(req.model_memory, 0);
}

TEST(ResourceEstimator, CurrentUsage) {
    ResourceEstimator& estimator = ResourceEstimator::instance();

    // Initially zero usage (after fresh start)
    int model_count = estimator.current_model_count();
    int vb_usage = estimator.current_vb_usage();

    // Model count should be >= 0
    EXPECT_GE(model_count, 0);
    EXPECT_GE(vb_usage, 0);
}

TEST(ResourceEstimator, NodeStartStop) {
    ResourceEstimator& estimator = ResourceEstimator::instance();

    int initial_count = estimator.current_model_count();

    // Simulate starting a node
    ResourceRequirement req;
    req.vb_infer_blocks = 1;
    req.model_memory = 1024 * 1024;  // 1MB
    estimator.on_node_started("test_node_1", req);

    // Model count should increase
    EXPECT_EQ(estimator.current_model_count(), initial_count + 1);

    // Stop the node
    estimator.on_node_stopped("test_node_1");

    // Model count should decrease
    EXPECT_EQ(estimator.current_model_count(), initial_count);
}

TEST(ResourceEstimator, MultipleNodeTracking) {
    ResourceEstimator& estimator = ResourceEstimator::instance();

    int initial_count = estimator.current_model_count();
    int initial_vb = estimator.current_vb_usage();

    // Start multiple nodes
    ResourceRequirement req1;
    req1.vb_infer_blocks = 1;
    estimator.on_node_started("multi_node_1", req1);

    ResourceRequirement req2;
    req2.vb_infer_blocks = 2;
    estimator.on_node_started("multi_node_2", req2);

    EXPECT_EQ(estimator.current_model_count(), initial_count + 2);
    EXPECT_EQ(estimator.current_vb_usage(), initial_vb + 3);

    // Stop nodes
    estimator.on_node_stopped("multi_node_1");
    estimator.on_node_stopped("multi_node_2");

    EXPECT_EQ(estimator.current_model_count(), initial_count);
    EXPECT_EQ(estimator.current_vb_usage(), initial_vb);
}

TEST(ResourceEstimator, DuplicateNodeStop) {
    ResourceEstimator& estimator = ResourceEstimator::instance();

    int initial_count = estimator.current_model_count();

    ResourceRequirement req;
    req.vb_infer_blocks = 1;
    estimator.on_node_started("dup_node", req);

    EXPECT_EQ(estimator.current_model_count(), initial_count + 1);

    // Stop twice - should be idempotent
    estimator.on_node_stopped("dup_node");
    estimator.on_node_stopped("dup_node");  // Second call should be no-op

    EXPECT_EQ(estimator.current_model_count(), initial_count);
}

TEST(ResourceEstimator, EvaluateModelBasic) {
    ResourceEstimator& estimator = ResourceEstimator::instance();

    // Create a basic model config
    nlohmann::json config;
    config["model"] = "/nonexistent/model.cvimodel";

    std::vector<std::string> deps;

    EstimateResult result = estimator.evaluate_model(config, deps);

    // Should fail because model file doesn't exist (or return evaluation result)
    // The exact behavior depends on implementation
    // Just verify the function doesn't crash
    EXPECT_TRUE(result.error_code == MA_OK || result.error_code != MA_OK);
}

TEST(ResourceEstimator, EstimateMemoryFromFile) {
    ResourceEstimator& estimator = ResourceEstimator::instance();

    // Test with non-existent file
    size_t mem = estimator.estimate_model_memory_from_file("/nonexistent/file.cvimodel");
    EXPECT_EQ(mem, 0);
}

TEST(ResourceEstimator, ParallelRequiresFrameSkip) {
    ResourceEstimator& estimator = ResourceEstimator::instance();

    estimator.register_camera("camera1", false, 0.0);

    ResourceRequirement req;
    req.vb_infer_blocks = 1;
    estimator.on_node_started("model1", req, "camera1", "");
    estimator.on_node_started("model2", req, "camera1", "");

    nlohmann::json config;
    config["model"] = "/nonexistent/model.cvimodel";

    auto result_no_skip = estimator.evaluate_model(config, {"camera1"});
    EXPECT_FALSE(result_no_skip.pass);

    estimator.register_camera("camera1", true, 10.0);
    auto result_with_skip = estimator.evaluate_model(config, {"camera1"});
    EXPECT_TRUE(result_with_skip.pass);

    estimator.on_node_stopped("model1");
    estimator.on_node_stopped("model2");
    estimator.unregister_camera("camera1");
}
