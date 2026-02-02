-- Benchmark sigmoid_max_with_argmax RVV optimization
local nn = lua_nn

print("=== RVV sigmoid_max Benchmark ===\n")

-- YOLO-like data shape: [80 classes, 8400 anchors]
local num_classes = 80
local num_anchors = 8400

-- Create random logits
local data = {}
for i = 1, num_classes * num_anchors do
    data[i] = math.random() * 20 - 10  -- Random values [-10, 10]
end

local tensor = nn.Tensor.new(data, {num_classes, num_anchors})
print(string.format("Input shape: [%d, %d] = %d elements",
    num_classes, num_anchors, num_classes * num_anchors))

-- Warmup
for i = 1, 3 do
    local _ = tensor:sigmoid_max_with_argmax(0)
end

-- Benchmark
local iterations = 10
local times = {}

for i = 1, iterations do
    local t0 = os.clock()
    local result = tensor:sigmoid_max_with_argmax(0)
    local elapsed = (os.clock() - t0) * 1000
    table.insert(times, elapsed)
end

-- Statistics
table.sort(times)
local sum = 0
for _, t in ipairs(times) do
    sum = sum + t
end
local mean = sum / #times
local median = times[math.ceil(#times / 2)]
local min = times[1]
local max = times[#times]

print(string.format("\nResults (%d iterations):", iterations))
print(string.format("  Mean:   %.2f ms", mean))
print(string.format("  Median: %.2f ms", median))
print(string.format("  Min:    %.2f ms", min))
print(string.format("  Max:    %.2f ms", max))

-- Verify correctness
print("\n=== Correctness Check ===")
local result = tensor:sigmoid_max_with_argmax(0)
local values_table = result.values:to_table()
print(string.format("Output values shape: [%d]", #values_table))
print(string.format("Output indices count: %d", #result.indices))
print(string.format("Sample values: [%.4f, %.4f, %.4f]",
    values_table[1], values_table[2], values_table[3]))
print(string.format("Sample indices: [%d, %d, %d]",
    result.indices[1], result.indices[2], result.indices[3]))

print("\n✓ Benchmark completed")
