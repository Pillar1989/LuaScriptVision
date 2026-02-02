-- Fine-grained scaling benchmark around 6400
local nn = lua_nn

print("=== Fine-grained Scaling Around 6400 ===\n")
print(string.format("%-10s %-12s %-15s", "Anchors", "Time (ms)", "Time/Anchor (μs)"))
print(string.format("%-10s %-12s %-15s", "-------", "----------", "--------------"))

local num_classes = 80
-- Test sizes around 6400
local anchor_counts = {5000, 5500, 6000, 6400, 6800, 7200, 7600, 8000, 8400}

for _, n_anchors in ipairs(anchor_counts) do
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
    local iterations = 3
    local total_time = 0
    for i = 1, iterations do
        local t0 = os.clock()
        local _ = tensor:sigmoid_max_with_argmax(0)
        total_time = total_time + (os.clock() - t0)
    end
    local mean_ms = (total_time / iterations) * 1000
    local us_per_anchor = mean_ms * 1000 / n_anchors
    local stride_kb = (n_anchors * 4) / 1024

    print(string.format("%-10d %-12.2f %-15.2f (stride: %.1f KB)",
        n_anchors, mean_ms, us_per_anchor, stride_kb))
end
