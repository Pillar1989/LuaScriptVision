-- YOLO11 Object Detection Script (Tensor API)
-- Supports both fused single-output and multi-head 6-output models
-- Optimized: filter by confidence first, then decode only valid candidates
local utils = lua_utils
local nn = lua_nn
local preprocess_lib = require("scripts.lib.preprocess")
local coco_labels = require("scripts.lib.coco")

local Model = {}

Model.config = {
    input_size = {640, 640},
    conf_thres = 0.25,
    iou_thres  = 0.45,
    stride     = 32,
    labels = coco_labels,
    reg_max = 16,
    strides = {8, 16, 32}
}

-- C++ Preprocess Configuration
Model.preprocess_config = {
    type = "letterbox",
    input_size = {640, 640},
    stride = 32,
    fill_value = 114
}

-- DFL decode for a SINGLE anchor point (efficient: only called for valid candidates)
-- Input: bbox_flat tensor [64, n_anchors], anchor index (0-based)
-- Output: l, t, r, b decoded distances
local function dfl_decode_single(bbox_flat, anchor_idx, reg_max)
    local l, t, r, b = 0, 0, 0, 0

    -- For each coordinate (l, t, r, b)
    for coord = 0, 3 do
        local start_ch = coord * reg_max

        -- Find max for numerical stability
        local max_val = -1e30
        for bin = 0, reg_max - 1 do
            local v = bbox_flat:at(start_ch + bin, anchor_idx)
            if v > max_val then max_val = v end
        end

        -- Softmax + weighted sum
        local sum_exp = 0
        local weighted = 0
        for bin = 0, reg_max - 1 do
            local v = bbox_flat:at(start_ch + bin, anchor_idx)
            local e = math.exp(v - max_val)
            sum_exp = sum_exp + e
            weighted = weighted + e * bin
        end

        local dist = weighted / sum_exp
        if coord == 0 then l = dist
        elseif coord == 1 then t = dist
        elseif coord == 2 then r = dist
        else b = dist end
    end

    return l, t, r, b
end

-- Process multi-head YOLO output (6 outputs: 3 bbox + 3 cls)
-- Optimized: filter by confidence FIRST, then decode only valid candidates
local function process_multi_head(outputs, meta)
    local reg_max = Model.config.reg_max
    local strides = Model.config.strides
    local conf_thres = Model.config.conf_thres

    local all_proposals = {}

    -- Process each scale (3 scales)
    for scale = 0, 2 do
        local bbox_key = "output" .. (scale * 2)
        local cls_key = "output" .. (scale * 2 + 1)

        local bbox_output = outputs[bbox_key]
        local cls_output = outputs[cls_key]

        if not bbox_output or not cls_output then
            print(string.format("Warning: Missing output at scale %d", scale))
            goto continue
        end

        local bbox_shape = bbox_output:shape()
        local cls_shape = cls_output:shape()
        local stride = strides[scale + 1]

        -- Get grid dimensions from shape [1, channels, H, W]
        local grid_h = bbox_shape[3]
        local grid_w = bbox_shape[4]
        local n_anchors = grid_h * grid_w

        -- Reshape to [channels, n_anchors]
        local bbox_flat = bbox_output:reshape({bbox_shape[2], n_anchors}):contiguous()
        local cls_flat = cls_output:reshape({cls_shape[2], n_anchors}):contiguous()

        -- Apply sigmoid to class scores and find max
        local cls_scores = cls_flat:sigmoid()
        local max_result = cls_scores:max_with_argmax(0)
        local max_scores = max_result.values
        local class_ids = max_result.indices

        -- FIRST: filter by confidence (this is very fast)
        local valid_indices = max_scores:where_indices(conf_thres, "ge")

        if #valid_indices == 0 then
            goto continue
        end

        -- THEN: decode DFL only for valid candidates (typically < 100)
        local filtered_scores = max_scores:index_select(0, valid_indices):to_table()

        for i, idx in ipairs(valid_indices) do
            -- Decode DFL for this single anchor
            local l, t, r, b = dfl_decode_single(bbox_flat, idx, reg_max)

            -- Calculate anchor center
            local row = math.floor(idx / grid_w)
            local col = idx % grid_w
            local ax = (col + 0.5) * stride
            local ay = (row + 0.5) * stride

            -- dist2bbox
            local x1 = ax - l * stride
            local y1 = ay - t * stride
            local x2 = ax + r * stride
            local y2 = ay + b * stride

            local cx = (x1 + x2) * 0.5
            local cy = (y1 + y2) * 0.5
            local w = x2 - x1
            local h = y2 - y1

            local cls_id = class_ids[idx + 1]
            local conf = filtered_scores[i]

            table.insert(all_proposals, {
                x = cx - w / 2.0,
                y = cy - h / 2.0,
                w = w,
                h = h,
                score = conf,
                class_id = cls_id,
                label = Model.config.labels[cls_id + 1] or "unknown"
            })
        end

        ::continue::
    end

    return all_proposals
end

-- Process fused single-output YOLO (output0 with shape [1, 84, 8400])
local function process_fused(outputs, meta)
    local output = outputs["output0"]

    -- YOLO11 fused format: [1, 84, 8400]
    -- First 4 channels: cx, cy, w, h (already decoded)
    -- Next 80 channels: class scores (already sigmoid-ed)
    local boxes = output:slice(1, 0, 4, 1):squeeze(0):contiguous()
    local scores = output:slice(1, 4, 84, 1):squeeze(0):contiguous()

    -- Find max class score and id
    local result = scores:max_with_argmax(0)
    local max_scores = result.values
    local class_ids = result.indices

    -- Filter by confidence
    local valid_indices = max_scores:where_indices(Model.config.conf_thres, "ge")

    if #valid_indices == 0 then
        return {}
    end

    -- Extract filtered boxes and scores
    local filtered_boxes = boxes:extract_columns(valid_indices)
    local filtered_scores = max_scores:index_select(0, valid_indices):to_table()

    local proposals = {}
    for i = 1, #valid_indices do
        local idx = valid_indices[i]
        local box_data = filtered_boxes[i]

        local cx = box_data[1]
        local cy = box_data[2]
        local w = box_data[3]
        local h = box_data[4]
        local cls_id = class_ids[idx + 1]
        local conf = filtered_scores[i]

        table.insert(proposals, {
            x = cx - w / 2.0,
            y = cy - h / 2.0,
            w = w,
            h = h,
            score = conf,
            class_id = cls_id,
            label = Model.config.labels[cls_id + 1] or "unknown"
        })
    end

    return proposals
end

-- Main postprocess function
function Model.postprocess(outputs, meta)
    local output_count = meta.output_count or 1

    local proposals
    if output_count == 6 then
        -- Multi-head YOLO (6 outputs: 3 bbox + 3 cls)
        print("[Postprocess] Multi-head YOLO (6 outputs)")
        proposals = process_multi_head(outputs, meta)
    else
        -- Fused single-output YOLO
        print("[Postprocess] Fused YOLO (1 output)")
        proposals = process_fused(outputs, meta)
    end

    -- Scale coordinates back to original image
    for _, box in ipairs(proposals) do
        box.x, box.y = preprocess_lib.scale_coords(box.x, box.y, meta)
        box.w = preprocess_lib.scale_size(box.w, meta)
        box.h = preprocess_lib.scale_size(box.h, meta)
    end

    -- Apply NMS
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)

    print(string.format("NMS final boxes: %d", #final_boxes))

    return final_boxes
end

return Model
