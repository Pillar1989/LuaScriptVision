-- Sigmoid_max scaling benchmark
local nn = lua_nn

print("=== sigmoid_max Scaling Benchmark ===\n")
print(string.format("%-10s %-12s %-15s", "Anchors", "Time (ms)", "Time/Anchor (μs)"))
print(string.format("%-10s %-12s %-15s", "-------", "----------", "--------------"))

local num_classes = 80
local anchor_counts = {400, 1600, 6400, 8400}

for _, n_anchors in ipairs(anchor_counts) do
    -- Create tensor
    local data = {}
    for i = 1, num_classes * n_anchors do
        data[i] = math.random() * 20 - 10
    end
    local tensor = nn.Tensor.new(data, {num_classes, n_anchors})

    -- Warmup
    for i = 1, 2 do
        local _ = tensor:sigmoid_max_with_argmax(0)
    end

    -- Measure
    local iterations = 5
    local total_time = 0
    for i = 1, iterations do
        local t0 = os.clock()
        local _ = tensor:sigmoid_max_with_argmax(0)
        total_time = total_time + (os.clock() - t0)
    end
    local mean_ms = (total_time / iterations) * 1000
    local us_per_anchor = mean_ms * 1000 / n_anchors

    print(string.format("%-10d %-12.2f %-15.2f", n_anchors, mean_ms, us_per_anchor))
end

print("\nConclusion: If time scaled linearly, Time/Anchor should be constant.")
