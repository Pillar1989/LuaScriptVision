-- Realistic sigmoid_max benchmark matching real YOLO pipeline data flow
local nn = lua_nn

print("=== Realistic sigmoid_max Benchmark (mimicking YOLO pipeline) ===\n")

-- Mimic Scale 0 from real pipeline: [1, 80, 80, 80] -> [80, 6400]
local batch = 1
local num_classes = 80
local grid_h, grid_w = 80, 80
local n_anchors = grid_h * grid_w  -- 6400

print(string.format("Creating tensor [%d, %d, %d, %d] = %d elements",
    batch, num_classes, grid_h, grid_w, batch * num_classes * grid_h * grid_w))

-- Create data in NCHW format (like TPU output)
local data = {}
for b = 1, batch do
    for c = 1, num_classes do
        for h = 1, grid_h do
            for w = 1, grid_w do
                data[#data + 1] = math.random() * 20 - 10
            end
        end
    end
end

-- Create tensor in NCHW format
local tensor_nchw = nn.Tensor.new(data, {batch, num_classes, grid_h, grid_w})
print(string.format("Created NCHW tensor, shape: [%s]", table.concat(tensor_nchw:shape(), "x")))

-- Reshape and make contiguous (like real pipeline)
print(string.format("\nReshaping to [%d, %d] and making contiguous...", num_classes, n_anchors))
local t_start = os.clock()
local tensor_flat = tensor_nchw:reshape({num_classes, n_anchors}):contiguous()
local t_reshape = (os.clock() - t_start) * 1000
print(string.format("Reshape+contiguous took: %.2f ms", t_reshape))
print(string.format("Reshaped tensor shape: [%s], is_contiguous: %s",
    table.concat(tensor_flat:shape(), "x"), tostring(tensor_flat:is_contiguous())))

-- Warmup
for i = 1, 3 do
    local _ = tensor_flat:sigmoid_max_with_argmax(0)
end

-- Benchmark
local iterations = 10
local times = {}

print(string.format("\nRunning %d iterations...", iterations))
for i = 1, iterations do
    local t0 = os.clock()
    local result = tensor_flat:sigmoid_max_with_argmax(0)
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

print("\n✓ Benchmark completed")
print(string.format("\nComparison with real pipeline Scale 0 (expected ~43ms)"))
