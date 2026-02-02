-- Direct [80, 6400] sigmoid_max benchmark
local nn = lua_nn

print("=== Direct [80, 6400] sigmoid_max Benchmark ===\n")

local num_classes = 80
local n_anchors = 6400

-- Create data directly in target shape
local data = {}
for i = 1, num_classes * n_anchors do
    data[i] = math.random() * 20 - 10
end

local tensor = nn.Tensor.new(data, {num_classes, n_anchors})
print(string.format("Created tensor shape: [%s], is_contiguous: %s",
    table.concat(tensor:shape(), "x"), tostring(tensor:is_contiguous())))

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

print(string.format("\nResults (%d iterations): Mean = %.2f ms", iterations, mean))
print(string.format("Expected (scaled from 8400): %.2f ms", 15.04 * 6400 / 8400))
print(string.format("Actual real pipeline: ~43 ms"))
