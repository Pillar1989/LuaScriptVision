# LuaScriptVision SSCMA Node Replacement Design

## Purpose
Replace sscma-node with LuaScriptVision while preserving full MQTT protocol compatibility and all node types, with no performance regression. Phase 1 prioritizes a zero-copy data plane for camera and JPEG inputs. Phase 2 adds multi-model pipelines per docs/architecture_future.

## Baseline and Constraints
- Baseline workload: sc530ai camera at 1080p, yolo11n 640x640, full chain (camera + stream + model + MQTT output).
- Performance: latency and FPS must be no worse than sscma-node under the baseline workload.
- Protocol: request/response/event/log frames and node lifecycle semantics must match sscma-node.
- Deployment: LuaScriptVision becomes the service, but does not need to keep the same binary or service name.
- Compatibility: allow optional protocol extensions and new node types, while keeping existing behavior unchanged.

## Phase 1: Data Plane First
- Inputs: camera and local JPEG files.
- JPEG path: VDEC decode to VPSS (MEM input) with zero-copy output.
- Camera path: VI to VPSS (ISP input) with zero-copy output.
- VPSS groups are separated by input type and never switched at runtime.
- Preprocess: Lua defines preprocess_config for common flows and granular ops (letterbox, resize, crop, pad, color conversion). C++ maps all preprocessing to VPSS channel parameters. If a combination cannot be expressed by VPSS, configuration fails with an error.
- Outputs: VPSS produces VB physical address buffers for TPU input. Tensor and DeviceBuffer carry physical address, alignment, stride, and lifecycle references.

## Control Plane and Nodes
- Reimplement NodeServer and NodeFactory in LuaScriptVision with full MQTT protocol compatibility.
- Node types: camera, model, stream, save, qrcode. All must support dynamic create, destroy, config, and dependencies.
- Concurrency: camera, stream, save, and model must run concurrently without impacting model throughput.

## Lua Script Contract and Model Outputs
- model node requires lua_script. algorithm is supported only for legacy clients and maps to default scripts defined in Lua.
- For sscma-node supported models, provided scripts must emit identical output fields and structures.
- For other models, output is defined by Lua and is passed through unmodified in MQTT data.

## Phase 2: Multi-Model Pipeline
- model node accepts pipeline config to run multiple models in serial or parallel modes.
- Shared FrameContext enables reuse of inputs, resource accounting, and synchronized output.
- Lua remains responsible for preprocess and postprocess, with C++ handling inference orchestration only.

## Error Handling and Verification
- Invalid preprocess combinations are rejected at config time.
- All builds require on-device tests and performance verification against the sscma-node baseline.
- Existing tests must pass; TPU-specific tests should skip when SDK is unavailable.
