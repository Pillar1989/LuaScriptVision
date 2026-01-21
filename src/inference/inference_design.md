# Inference Design (SG200X CVI Runtime + ONNX Runtime)

## 1. Scope and Goals

This document defines the inference layer design for:
- General platforms using ONNX Runtime.
- SG200X platforms using CVI Runtime (TPU).

Goals:
- Provide a single InferenceSession interface that supports multiple backends.
- Enable cpp_main validation on both ONNX and CVI runtime.
- Preserve future expansion to full hardware pipelines with zero-copy.
- Align with future architecture documents (Pipeline, SessionManager, Resource Management).

Non-goals:
- No code examples or API usage snippets in this document.
- No implementation details tied to a single board or demo model.

## 2. Terminology

| Term | Meaning |
|------|---------|
| Backend | Inference engine (ONNX Runtime or CVI Runtime). |
| Session | A loaded model instance with fixed input/output descriptors. |
| Frame | Image container from CV modules (camera or decoder). |
| Tensor | Inference input/output data representation. |
| DeviceBuffer | Memory buffer with device type and optional physical address. |
| Zero-copy | Using physical address memory without CPU memcpy. |
| Physical address mode | TPU input points to physical memory (VB/ION). |
| VB pool | CVI video buffer pool for DMA-capable memory. |
| ION | CVI DMA memory allocation and cache control. |

## 3. InferenceSession Interface

The inference layer exposes a backend-agnostic Session interface. The interface is described by the capabilities below, not by code.

### 3.1 Core Responsibilities

- Model loading and lifetime management.
- Input/output descriptor discovery (shape, dtype, layout, quantization).
- Execution of inference with consistent error reporting.
- Buffer compatibility checks and conversion pathways.
- Optional zero-copy path when a physical address is available.

### 3.2 Required Interface Surface

| Interface Item | Description |
|----------------|-------------|
| Backend identity | Returns backend name (for logging and test output). |
| Input descriptors | Shape, dtype, layout, quant params, alignment. |
| Output descriptors | Shape, dtype, layout, quant params, alignment. |
| Run inference (high-level) | run(): Accepts Tensor inputs, returns output Tensor(s). |
| Run inference (low-level) | run_raw(): Accepts DeviceBuffer pointers for zero-copy pipelines. |
| Device support | Reports supported device types and zero-copy capability. |
| Profiling hooks | Optional timing stats for preprocess/infer/postprocess. |

### 3.3 Descriptor Fields

Input and output descriptors must include:
- Shape and element count.
- Data type (float32, FP16/BF16, int8, uint8).
- Layout (NCHW or NHWC as defined by model).
- Quantization parameters (scale and zero_point if quantized).
- Required alignment (TPU requires 64-byte alignment).
- Acceptable memory types (CPU, TPU, or DMA).
- Strides per dimension (for non-contiguous tensor views).

Note: Video frame properties (pixel format, plane layout, row stride) are described by FrameDescriptor (Section 13A), not TensorDescriptor.

### 3.4 Ownership Rules

- Session owns model metadata and runtime handles.
- Caller owns input Tensor and receives output Tensor ownership.
- Zero-copy inputs must not be freed until inference completes.
- Output buffers are either owned by the Session (pool) or by the caller, but never both.

**Output buffer modes:**

| Mode | Ownership | Lifetime | Use Case |
|------|-----------|----------|----------|
| Caller-allocated | Caller | Until caller frees | run_raw() with pre-allocated buffers |
| Session-pooled | Session | Until next run() call | Default run() behavior |
| Caller-returned | Caller | Until caller frees | run() returns new tensors |

**Session-pooled output lifetime:**
- Session maintains internal output buffer pool.
- Output tensors from run() are valid only until the next run() call.
- Caller must copy output data if persistence beyond next inference is needed.
- Reduces allocation overhead for streaming/video inference.

### 3.5 Cache Coherency Rules

For DMA memory (VB/ION) shared between CPU and hardware:

| Operation | Before | After |
|-----------|--------|-------|
| CPU writes → TPU reads | flush_cache() | - |
| TPU writes → CPU reads | - | invalidate_cache() |
| CPU writes → VPSS reads | flush_cache() | - |
| VPSS writes → CPU reads | - | invalidate_cache() |

**Session responsibilities:**
- CviSession handles cache operations internally for TPU memory.
- For DMA memory inputs (VbMemory/IonMemory), caller must ensure cache coherency before passing to Session.
- For DMA memory outputs, Session invalidates cache before returning to caller.

## 4. Data Path and Memory Model

### 4.1 CPU Path (ONNX Runtime)

Text diagram:

```
Input image (CPU)
  -> CPU preprocess (OpenCV)
  -> Tensor (CPU)
  -> ONNX Runtime
  -> Output Tensor (CPU)
  -> CPU postprocess
```

### 4.2 TPU Path with CPU Preprocess (cpp_main baseline)

Text diagram:

```
Input image (CPU)
  -> CPU preprocess (OpenCV)
  -> Quantize or cast (CPU)
  -> TPU memory (CVI Runtime allocation)
  -> CVI Runtime inference
  -> Output in TPU memory
  -> Dequantize to CPU (only if CPU postprocess)
```

This path does not require VB or ION. It is the minimum path to validate TPU inference in cpp_main.

### 4.3 Full Hardware Path (Camera)

Text diagram:

```
Camera
  -> VI -> ISP
  -> VPSS (resize / cvtColor / crop)
  -> VB pool (physical memory)
  -> CVI Runtime inference (physical address mode)
  -> Output in TPU memory
  -> CPU postprocess or follow-on TPU pipeline
```

Notes:
- VPSS input mode is fixed per group (ISP input group is separate from MEM input group).
- Binding is used for control plane even when data path is direct in ISP mode.
- Physical address must be obtained via official CVI APIs only.

### 4.4 Full Hardware Path (JPEG Decode)

Text diagram:

```
JPEG
  -> VDEC
  -> VB pool
  -> VPSS (resize / cvtColor / crop)
  -> VB pool (physical memory)
  -> CVI Runtime inference (physical address mode)
```

ION may be used for intermediate buffers, but VPSS and VDEC frames must be backed by VB pools.

## 5. Quantization and Layout Handling

- Session must read model-defined dtype and layout for both input and output.
- For INT8 or UINT8, quant params must be stored with the descriptor and exposed to postprocess.
- When output is quantized, dequantization is required only if CPU postprocess expects float.
- Layout conversions are allowed but must be explicit and measurable in performance logs.

## 6. Zero-copy Conditions

Zero-copy is allowed only when all conditions are true:
- Input is backed by CVI-managed physical memory (VB or ION).
- Physical address is provided by CVI runtime or CVI MPI APIs. **Physical addresses MUST NOT be manually calculated; only use addresses returned by official CVI APIs.**
- Alignment and stride requirements are satisfied (64-byte alignment on SG200X).
- Memory ownership is clearly managed until inference completes.
- Cache coherency is maintained:
  - Before TPU read: `CVI_SYS_IonFlushCache()` to flush CPU writes to memory.
  - After TPU write: `CVI_SYS_IonInvalidateCache()` to invalidate CPU cache before CPU reads.

If any condition is not met, the Session must fall back to a copy path.

## 7. Buffer Management

### 7.1 TPU Buffer Pool

- A TPU memory pool is maintained by the CVI backend for reuse.
- Pool entries are keyed by size, dtype, and layout.
- Pool allocation must be compatible with CVI Runtime alignment requirements.

### 7.2 VB Pool Planning

- VB pools are defined in a single place (MmfContext plan).
- Pools must cover:
  - Camera output frames.
  - VPSS intermediate frames.
  - VDEC output frames.
  - Model input sizes.
- Pool sizes must be computed with official CVI buffer size APIs.

## 8. Error Handling and Observability

- All CVI Runtime calls must check return codes and log the function name and error code.
- Errors are classified as:
  - Configuration errors (bad model, unsupported dtype).
  - Resource errors (no VB block, no TPU memory).
  - Runtime errors (inference failure).
- When hardware path fails, the system may fall back to CPU inference if enabled.

## 9. Build-time Selection

- Use build-time macros to enable the backend:
  - ONNX Runtime for general platforms.
  - CVI Runtime when SG200X SDK is available.
- The same cpp_main entry point must build in both configurations.

## 10. cpp_main Validation Strategy

For cpp_main:
- ONNX path validates CPU inference and baseline correctness.
- CVI path validates TPU inference without VB/ION dependency.
- Hardware pipeline validation is staged separately with camera or decoder inputs.

## 11. Future Integration Points

This design is aligned with the future architecture documents:
- SessionManager can pool multiple InferenceSessions (up to 3).
- Pipeline can schedule Session runs in serial or parallel modes.
- Frame-level pipeline can reuse the same buffer pool and zero-copy paths.

Text diagram:

```
Pipeline (serial/parallel)
  -> SessionManager (max 3 sessions)
     -> InferenceSession (ONNX or CVI)
        -> BufferManager (CPU/TPU/VB/ION)
```

---

## 12. Type Definitions

### 12.1 DType (Data Type Enumeration)

Defines supported data types for inference tensors.

| Value | Description | Size (bytes) | Use Case |
|-------|-------------|--------------|----------|
| FLOAT32 | 32-bit floating point | 4 | ONNX default, CPU postprocess |
| FP16 | 16-bit floating point (IEEE 754) | 2 | Some ONNX models (per SDK support) |
| BF16 | 16-bit brain floating point | 2 | CVI models (per SDK support) |
| INT8 | 8-bit signed integer | 1 | CVI quantized models |
| UINT8 | 8-bit unsigned integer | 1 | CVI quantized models |
| INT32 | 32-bit signed integer | 4 | Indices, labels |

Note: FP16 vs BF16 availability depends on SDK support. Check official SDK headers for platform-specific support.

Helper functions:
- `dtype_size(DType) -> size_t`: Returns bytes per element.
- `dtype_name(DType) -> const char*`: Returns string representation.
- `is_quantized(DType) -> bool`: Returns true for INT8/UINT8.

### 12.2 Layout (Memory Layout Enumeration)

Defines tensor memory layout conventions.

| Value | Description | Dimension Order | Use Case |
|-------|-------------|-----------------|----------|
| NCHW | Channel-first | Batch, Channel, Height, Width | ONNX default |
| NHWC | Channel-last | Batch, Height, Width, Channel | TFLite, some NPU |
| NC | Feature vector | Batch, Channel | Classification output |
| UNKNOWN | Not determined | - | Dynamic shapes |

Helper functions:
- `layout_name(Layout) -> const char*`: Returns string representation.
- `infer_layout(shape) -> Layout`: Heuristic layout detection.

### 12.3 QuantParams (Quantization Parameters)

Stores quantization scale and zero point for INT8/UINT8 tensors.

Fields:
- `scale`: float (default 1.0)
- `zero_point`: int32_t (default 0)

Semantics:
```
float_value = (quantized_value - zero_point) * scale
quantized_value = round(float_value / scale) + zero_point
```

Methods:
- `is_identity() -> bool`: Returns true if scale==1.0 and zero_point==0.
- `dequantize(int8_t) -> float`: Converts quantized to float.
- `quantize(float) -> int8_t`: Converts float to quantized.

Note: Per-channel quantization is reserved for future extension.

---

## 13. TensorDescriptor Structure

Describes the properties of an inference tensor (model input or output). This is distinct from FrameDescriptor (Section 13A) which describes video frame properties.

### 13.1 Fields

| Field | Type | Description |
|-------|------|-------------|
| name | string | Tensor name from model |
| shape | vector\<int64_t\> | Dimension sizes |
| strides | vector\<int64_t\> | Stride per dimension in elements (for non-contiguous access) |
| dtype | DType | Data type |
| layout | Layout | Memory layout (NCHW, NHWC) |
| quant_params | QuantParams | Quantization parameters (if quantized) |
| alignment | size_t | Required byte alignment (default 64) |
| supported_devices | set\<DeviceType\> | Compatible device types (CPU, TPU, DMA) |

### 13.2 Derived Properties

- `element_count() -> int64_t`: Product of all shape dimensions.
- `size_bytes() -> size_t`: element_count × dtype_size.
- `is_quantized() -> bool`: True if dtype is INT8 or UINT8.

### 13.3 Compatibility Checks

- `is_compatible(Tensor&) -> bool`: Checks if tensor matches this descriptor.
- `is_compatible(DeviceBuffer&) -> bool`: Checks if buffer can hold this tensor.

Compatibility requires:
- Shape matches (or is broadcastable).
- DType matches.
- Alignment is satisfied.
- Device type is in supported_devices.

---

## 13A. FrameDescriptor Structure

Describes the properties of a video frame from VPSS/VDEC/VI. This is distinct from TensorDescriptor which describes model tensor properties.

### 13A.1 Fields

| Field | Type | Description |
|-------|------|-------------|
| width | uint32_t | Frame width in pixels |
| height | uint32_t | Frame height in pixels |
| pixel_format | PixelFormat | CVI pixel format (NV12, NV21, RGB, etc.) |
| plane_count | size_t | Number of planes (1 for packed, 2-3 for planar) |
| plane_strides | vector\<uint32_t\> | Stride (bytes per row) for each plane |
| plane_offsets | vector\<size_t\> | Byte offset for each plane from base address |
| physical_address | uint64_t | Physical address (0 if CPU-only) |
| device | DeviceType | Memory device type (CPU or DMA) |

### 13A.2 Pixel Formats

| Format | Planes | Description |
|--------|--------|-------------|
| NV12 | 2 | Y plane + interleaved UV plane |
| NV21 | 2 | Y plane + interleaved VU plane |
| YUV420P | 3 | Y + U + V separate planes |
| RGB_PACKED | 1 | Interleaved RGB (3 bytes/pixel) |
| BGR_PACKED | 1 | Interleaved BGR (3 bytes/pixel) |

### 13A.3 Stride Semantics

Hardware outputs (VPSS/VDEC) often have stride != width × bytes_per_pixel due to alignment requirements.

| Source | Typical Stride |
|--------|----------------|
| VPSS output | Aligned to 64 bytes |
| VDEC output | Aligned to 64 bytes |
| CPU allocation | width × bytes_per_pixel |

Stride must be checked during Frame→Tensor conversion to avoid data corruption.

---

## 14. InferenceSession Interface

Abstract interface for backend-agnostic model inference.

### 14.1 Backend Identity

| Method | Return | Description |
|--------|--------|-------------|
| backend_name() | const char* | Returns "onnxruntime" or "cviruntime" |

### 14.2 Model Information

| Method | Return | Description |
|--------|--------|-------------|
| num_inputs() | size_t | Number of model inputs |
| num_outputs() | size_t | Number of model outputs |
| input_descriptor(idx) | const TensorDescriptor& | Descriptor for input at index |
| output_descriptor(idx) | const TensorDescriptor& | Descriptor for output at index |
| input_name(idx) | const string& | Name of input at index |
| output_name(idx) | const string& | Name of output at index |

### 14.3 Device Support

| Method | Return | Description |
|--------|--------|-------------|
| supports_device(DeviceType) | bool | True if session accepts input from device |
| supports_zero_copy() | bool | True if physical address mode available |
| preferred_device() | DeviceType | Optimal input device type |

### 14.4 Inference Execution

| Method | Parameters | Return | Description |
|--------|------------|--------|-------------|
| run | vector\<Tensor\>& inputs | vector\<Tensor\> | Execute inference, return outputs |
| run_raw | inputs: DeviceBuffer**, outputs: DeviceBuffer**, count | InferenceResult | Low-level zero-copy interface |

#### run() Semantics

- Accepts Tensor inputs (any device type: CPU, TPU, or DMA).
- Converts inputs to preferred device if needed.
- Executes model inference.
- Returns output tensors (session-pooled, valid until next run()).
- Output tensor device depends on backend.
- Throws on error (invalid input, inference failure).

**Input requirements:**
- Input count must match num_inputs().
- Each input must be compatible with corresponding input_descriptor().
- For DMA inputs, caller must ensure cache coherency before call.

#### run_raw() Semantics

- Caller provides pre-allocated input and output buffers via DeviceBuffer pointers.
- Used for zero-copy hardware pipelines.
- Returns InferenceResult with status and optional error info.
- Caller manages buffer lifecycle.
- Does not throw; error handling via return value.

**InferenceResult:**

| Field | Type | Description |
|-------|------|-------------|
| success | bool | True if inference completed |
| error_code | int | Backend-specific error code (0 = success) |
| error_message | string | Human-readable error description |

**run_raw() error codes:**

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | Input buffer incompatible (size, alignment, device) |
| 2 | Output buffer incompatible |
| 3 | Model execution failed |
| 4 | Memory allocation failed |

### 14.5 Thread Safety

| Aspect | Guarantee |
|--------|-----------|
| Multiple sessions | Different sessions can run concurrently |
| Single session | NOT thread-safe; do not call run() from multiple threads |
| Reentrancy | NOT reentrant; do not call run() from within callbacks |

**Rationale:**
- TPU hardware is a shared resource; concurrent access requires explicit scheduling.
- Session maintains internal state (output buffers) that is not thread-safe.
- For parallel pipelines, use separate sessions or external synchronization.

### 14.6 Profiling (Optional)

| Method | Parameters | Return | Description |
|--------|------------|--------|-------------|
| enable_profiling | bool enable | void | Enable/disable timing collection |
| get_timing_stats | - | TimingStats | Returns timing breakdown |

TimingStats contains:
- `preprocess_ms`: Time for input preparation.
- `inference_ms`: Time for model execution.
- `postprocess_ms`: Time for output handling.

---

## 15. OnnxSession Specification

OnnxSession implements InferenceSession for ONNX Runtime backend.

### 15.1 Construction

| Parameter | Type | Description |
|-----------|------|-------------|
| model_path | string | Path to .onnx file |
| num_threads | int | Inference thread count (default 4) |

Construction performs:
- Creates Ort::Env with warning log level.
- Creates Ort::Session with ORT_ENABLE_ALL optimization.
- Extracts input/output descriptors from model metadata.
- Stores input/output names.

### 15.2 Device Support

| Method | Return | Reason |
|--------|--------|--------|
| supports_device(CPU) | true | ONNX Runtime uses CPU |
| supports_device(TPU) | false | Not supported |
| supports_device(NPU) | false | Not supported |
| supports_zero_copy() | false | CPU memory only |
| preferred_device() | CPU | Only option |

### 15.3 Inference Behavior

run() performs:
1. If input tensor is not on CPU, copy to CPU.
2. If model expects Float16, convert input.
3. Execute Ort::Session::Run().
4. If model outputs Float16, convert to Float32.
5. Return output tensors on CPU.

### 15.4 Descriptor Extraction

Input/output descriptors are populated from ONNX model:
- Shape from GetTensorTypeAndShapeInfo().
- DType from GetElementType() mapping.
- Layout inferred from shape dimensions.
- quant_params defaults to identity (no quantization).
- alignment defaults to 64 bytes.
- supported_devices contains only CPU.

---

## 16. CviSession Specification

CviSession implements InferenceSession for CVI Runtime (TPU) backend.

### 16.1 Construction

| Parameter | Type | Description |
|-----------|------|-------------|
| model_path | string | Path to .cvimodel file |

Construction performs:
- Calls CVI_NN_RegisterModel() to load model.
- Retrieves input/output tensors via CVI_NN_GetInputOutputTensors().
- Extracts descriptors from CVI_TENSOR structures.
- Stores quantization parameters from model.

### 16.2 Device Support

| Method | Return | Reason |
|--------|--------|--------|
| supports_device(CPU) | true | Can accept CPU input (with copy) |
| supports_device(TPU) | true | Native TPU memory |
| supports_zero_copy() | true | When physical address available |
| preferred_device() | TPU | Best performance |

### 16.3 Inference Behavior

run() performs:
1. Validate input compatibility (shape, dtype).
2. If input is CPU tensor:
   - Allocate TPU memory (or reuse from pool).
   - Quantize if model expects INT8/UINT8.
   - Copy data to TPU memory.
3. If input has physical address (VB/ION):
   - Configure zero-copy mode.
   - Set physical address in CVI_TENSOR.
4. Call CVI_NN_Forward().
5. Create output tensors from CVI_TENSOR.
6. If output is quantized and caller needs float:
   - Dequantize using output quant_params.
7. Return output tensors.

### 16.4 Descriptor Extraction

Input/output descriptors are populated from CVI_TENSOR:
- Shape from tensor.shape.dim[].
- DType from tensor.fmt (CVI_FMT_INT8, CVI_FMT_FP32, etc.).
- Layout: NCHW (CVI default).
- quant_params from tensor.qscale.
- alignment: 64 bytes (TPU requirement).
- supported_devices: {CPU, TPU}.

### 16.5 CVI-Specific Extensions

| Method | Parameters | Return | Description |
|--------|------------|--------|-------------|
| set_physical_address_mode | bool enable | void | Enable zero-copy with physical addresses |
| get_input_tensor | - | CVI_TENSOR* | Direct access for zero-copy setup |
| get_output_tensor | - | CVI_TENSOR* | Direct access to output |

---

## 17. DeviceBuffer Extensions

Extensions to the existing tensor/DeviceBuffer interface for inference integration.

### 17.1 Memory Type Hierarchy

| DeviceType | Description | Zero-copy to TPU | Cache Management |
|------------|-------------|------------------|------------------|
| CPU | Standard CPU heap memory | No | N/A |
| TPU | CVI Runtime allocated memory | Yes | CVI Runtime handles |
| DMA | VB/ION physical memory | Yes | Explicit flush/invalidate required |

Note: DMA memory (VB pool or ION allocation) is accessible by both CPU and hardware accelerators (VPSS, TPU). It requires explicit cache management when shared between CPU and hardware.

### 17.2 TpuMemory

TpuMemory extends DeviceBuffer for CVI Runtime allocated memory.

| Method | Return | Description |
|--------|--------|-------------|
| device() | DeviceType::TPU | Device type |
| physical_address() | uint64_t | Physical address for zero-copy |
| is_aligned(size_t) | bool | Check alignment |

Memory source: Allocated via CVI Runtime memory APIs. Cache coherency managed internally by CVI Runtime.

### 17.3 VbMemory

VbMemory extends DeviceBuffer for VB pool backed memory.

| Field | Type | Description |
|-------|------|-------------|
| vb_block_ | VB_BLK | VB block handle |
| pool_id_ | VB_POOL | Source pool ID |

| Method | Return | Description |
|--------|--------|-------------|
| device() | DeviceType::DMA | DMA-capable memory |
| physical_address() | uint64_t | From VIDEO_FRAME_INFO_S |
| flush_cache() | void | Flush CPU writes before hardware access |
| invalidate_cache() | void | Invalidate CPU cache after hardware writes |
| release() | void | Return block to pool |

Memory source: Acquired from VB pool via CVI_VB_GetBlock().

### 17.4 IonMemory

IonMemory extends DeviceBuffer for ION allocated memory.

| Method | Return | Description |
|--------|--------|-------------|
| device() | DeviceType::DMA | DMA-capable memory |
| physical_address() | uint64_t | From CVI_SYS_IonAlloc |
| flush_cache() | void | CVI_SYS_IonFlushCache |
| invalidate_cache() | void | CVI_SYS_IonInvalidateCache |

Memory source: Allocated via CVI_SYS_IonAlloc().

---

## 18. Frame-Tensor Integration

Defines conversion between cv/ module Frame (described by FrameDescriptor) and inference Tensor (described by TensorDescriptor).

### 18.1 Conversion Overview

Frame and Tensor serve different purposes:

| Aspect | Frame (FrameDescriptor) | Tensor (TensorDescriptor) |
|--------|-------------------------|---------------------------|
| Purpose | Video frame from hardware | Model input/output |
| Layout | Planar (NV12, YUV420P) | Channel-first/last (NCHW, NHWC) |
| Stride | Per-plane row stride (bytes) | Per-dimension element stride |
| Format | Pixel format (NV12, RGB) | Data type (float32, int8) |

### 18.2 Frame to Tensor Conversion

Conversion requires both descriptors:
- Input: Frame with FrameDescriptor (pixel format, plane strides)
- Target: TensorDescriptor (shape, dtype, layout)

**Conversion steps:**

1. Validate pixel format compatibility (e.g., RGB → NCHW float32).
2. Handle stride mismatch: if frame stride != tensor expected stride, copy with stride correction.
3. Handle format conversion: NV12/NV21 → RGB requires color space conversion.
4. Handle data type: uint8 pixel → float32 tensor requires normalization.
5. Handle layout: HWC frame data → NCHW tensor requires transpose.

**Zero-copy conditions:**

Zero-copy is possible only when:
- Frame is DMA-backed (has physical address).
- Pixel format matches tensor expectation (no conversion needed).
- Stride matches or tensor supports stride-based access.
- Alignment requirements satisfied.

### 18.3 Tensor to Frame Conversion

Used for visualization or chained processing:

1. If tensor is on TPU: Copy to CPU first.
2. Determine appropriate pixel format from tensor layout.
3. Create Frame with matching FrameDescriptor.
4. Handle denormalization if needed (float → uint8).

### 18.4 Physical Address Access

Physical addresses are obtained from FrameDescriptor, never calculated:
- DMA-backed frame: Use `physical_address` field from FrameDescriptor.
- CPU-backed frame: No physical address (must copy to DMA memory for zero-copy).

---

## 19. Session Factory

Creates appropriate InferenceSession based on model format.

### 19.1 Factory Function

Function: `create_session(model_path) -> unique_ptr<InferenceSession>`

Logic:
- If model_path ends with ".cvimodel" and USE_CVI_TPU defined:
  - Return CviSession instance.
- If model_path ends with ".onnx" and USE_ONNX_RUNTIME defined:
  - Return OnnxSession instance.
- Otherwise: Throw "Unsupported model format" error.

### 19.2 Build-time Configuration

| Macro | Effect |
|-------|--------|
| USE_ONNX_RUNTIME | OnnxSession available |
| USE_CVI_TPU | CviSession available (SG200X platform) |

Both macros can be defined simultaneously for platforms supporting both backends.

---

## 20. File Organization

```
src/inference/
├── dtype.h                    # DType enumeration
├── layout.h                   # Layout enumeration
├── quant_params.h             # QuantParams structure
├── tensor_descriptor.h        # TensorDescriptor structure
├── inference_session.h        # Abstract InferenceSession interface
├── onnx_session.h             # OnnxSession declaration
├── onnx_session.cpp           # OnnxSession implementation
├── cvi_session.h              # CviSession declaration
├── cvi_session.cpp            # CviSession implementation
└── session_factory.h          # create_session() factory

src/modules/tensor/
├── tpu_memory.h               # TpuMemory declaration
├── tpu_memory.cpp             # TpuMemory implementation
├── vb_memory.h                # VbMemory declaration
└── vb_memory.cpp              # VbMemory implementation

src/modules/cv/
├── frame_tensor.h             # Frame ↔ Tensor conversion
└── frame_tensor.cpp           # Frame ↔ Tensor implementation
```

