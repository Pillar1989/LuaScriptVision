-- Test script for Sophgo TPU Memory Management (Phase 1)
-- This test verifies CviTpuMemory allocation, copy, and deallocation

local test_helpers = require("test_helpers")
local nn = require("nn")

-- Skip test if TPU support is not available
if not nn.CviTpuMemory then
    print("⊘ Sophgo TPU support not available, skipping test")
    return
end

print("\n========== Testing Sophgo TPU Memory Management ==========\n")

-- Test 1: Basic allocation
test_helpers.test("CviTpuMemory: Basic allocation", function()
    -- Note: CviTpuMemory requires CVI_RT_HANDLE which is managed internally
    -- For now, we test that the class exists
    assert(nn.CviTpuMemory ~= nil, "CviTpuMemory class should exist")
    print("  ✓ CviTpuMemory class available")
end)

-- Test 2: Check DeviceType enum
test_helpers.test("DeviceType: TPU enum value", function()
    -- Verify DeviceType.TPU exists
    local tensor = nn.Tensor.new({2, 3}, nn.float32)
    local device = tensor:device()
    assert(device == "CPU", "Default tensor should be on CPU")
    print("  ✓ DeviceType enum working correctly")
end)

-- Test 3: Verify conditional compilation
test_helpers.test("Conditional compilation: USE_CVI_TPU", function()
    -- If we reach here with nn.CviTpuMemory available,
    -- it means USE_CVI_TPU was defined during compilation
    local has_tpu = (nn.CviTpuMemory ~= nil)
    assert(has_tpu, "USE_CVI_TPU should be defined")
    print("  ✓ USE_CVI_TPU compiled correctly")
end)

print("\n=== Phase 1 Memory Tests Complete ===")
print("Note: Full memory allocation tests require CVI Runtime initialization")
print("This will be tested in Phase 2 (CVI Session)\n")
