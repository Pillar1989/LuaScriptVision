# CV Module Zero-Copy Architecture Design

## 1. 设计目标

实现基于 SG200X 硬件加速的计算机视觉 pipeline，支持 Camera 零拷贝和文件硬件加速。

### 核心特性
- 多路径支持：Camera 零拷贝 / 文件硬件加速 / CPU 软件
- 智能路径选择：根据输入类型自动优化
- 统一 API：用户无需关心底层实现
- 容错机制：硬件失败时自动降级

## 2. 资源分配策略

### VI (Video Input) - Camera输入
- **VI_DEV**: 0
- **VI_PIPE**: 0
- **VI_CHN**: 0
- **Sensor**: OV5647/GC2053/SC530AI (自动检测)
- **Resolution**: 1920x1080 (默认，可配置)
- **Frame Rate**: 30 fps (默认，可配置)
- **Pixel Format**: NV21 (default) / RGB888

### VPSS (Video Post-Processing) - AI预处理
- **VPSS_GRP**: 0
- **VPSS_CHN**: 0 (AI推理预处理专用)
- **功能**:
  - Resize (任意分辨率)
  - Format conversion (NV21↔RGB888)
  - Crop (可选)
  - Normalize (可选)

### VENC (Video Encoder) - 不使用
- LuaScriptVision专注AI推理，不需要视频编码

### VB (Video Buffer) Pool
- **Pool配置**: 8MB × 3 blocks
- 足够容纳 1920×1080×3 (RGB888) 帧
- 用于VI/VPSS零拷贝传输

## 3. 架构设计

### 3.1 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                     Lua API Layer                           │
│  cv.Camera()  cv.resize()  cv.cvtColor()  etc.             │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                     Frame (Unified Frame)                  │
│  - cv::Mat storage (CPU path)                               │
│  - VIDEO_FRAME_INFO_S storage (zero-copy path)             │
│  - Physical address tracking                                │
│  - Lazy conversion                                          │
└────────────┬────────────────────────────────────────────────┘
             │
       ┌─────┴─────┐
       │           │
┌──────▼──────┐ ┌─▼──────────────┐
│  CPU Path   │ │ Zero-Copy Path │
│  (OpenCV)   │ │  (VI+VPSS)     │
└─────────────┘ └────────────────┘
```

### 3.2 数据流路径设计

#### 路径 A：Camera 零拷贝路径

```
Camera Sensor (OV5647/GC2053)
    ↓ MIPI CSI-2
VI Module (DEV=0, PIPE=0, CHN=0)
    ↓ VIDEO_FRAME_INFO_S (物理地址)
    ↓ Zero-copy binding
VPSS Module (GRP=1, CHN=0)
    ↓ Hardware resize/convert
VIDEO_FRAME_INFO_S (物理地址)
    ↓ Zero-copy
TPU Inference
```

**特性**：
- 全程使用物理地址
- VI → VPSS → TPU 无内存拷贝
- VB pool 循环使用

#### 路径 B：文件硬件加速路径（全链路优化）✅

```
JPEG File
    ↓ File read (0.3ms)
JPEG data in memory
    ↓ VDEC Hardware Decode (3.9ms, 阻塞模式)
VIDEO_FRAME_INFO_S (NV21, 物理地址)
    ↓ Zero-copy
VPSS Module (GRP=1, CHN=0)
    ↓ Hardware resize/convert (7-8ms)
VIDEO_FRAME_INFO_S (物理地址)
    ↓ Zero-copy
TPU Inference
```

**性能数据（1280x720 JPEG）**：
- JPEG decode (HW VDEC sync): **3.91 ms** (vs 134ms OpenCV, 34.4x faster)
- VPSS resize (1080p→640x640): **7.7 ms**
- 全流程（file read + decode + VPSS): **~12 ms** (vs 145ms CPU, 12x faster)

**优化亮点**：
- ✅ **IonImageLoader**: 统一 VB pool 管理（避免 Ion 分配冲突）
- ✅ **HwJpegDecoder sync mode**: 阻塞 API，无线程同步延迟（51ms→3.9ms, 13x提升）
- ✅ **零拷贝链路**: VDEC → VPSS 全程物理地址传递
- ✅ **VB pool 统一**: IonImageLoader、VDEC、VPSS 共享公共 VB pool

#### 路径 C：CPU 纯软件路径

```
File / Memory
    ↓ cv::imread()
cv::Mat (CPU memory)
    ↓ cv::resize (CPU software)
cv::Mat
    ↓ Session::run (ONNX Runtime CPU)
```

**特性**：
- 纯 OpenCV 实现
- 无硬件依赖
- 跨平台兼容

### 3.4 文件组织

```
src/modules/cv/
├── cv_design.md                 # 本设计文档
├── cv_types.h                   # 公共类型定义
├── cvi_frame.h/cpp              # 统一Frame抽象（含CVI特定代码）
├── cv_camera.h/cpp              # Camera接口
├── cvi_camera.h/cpp             # CVI硬件Camera实现 (VI)
├── cv_processor.h/cpp           # 图像处理接口
├── cvi_vpss_processor.h/cpp     # CVI VPSS硬件处理器
├── opencv_processor.h/cpp       # OpenCV CPU处理器
├── cv_helpers.h/cpp             # 智能后端选择层
├── ion_image_loader.h/cpp       # VB pool统一图像加载器 ✅ NEW
└── hw_jpeg_decoder.h/cpp        # VDEC硬件JPEG解码器 ✅ NEW
```

**文件命名规范**：
- `cv_*`: 公共接口和类型，平台无关
- `cvi_*`: CVI平台特有实现或包含CVI特定代码（条件编译）
- `opencv_*`: OpenCV CPU实现
- `hw_*`: 硬件加速器封装（VDEC等）
- `ion_*`: 内存管理相关（VB pool/Ion）

## 4. 关键类设计

### 4.1 Frame - 统一帧表示

```cpp
class Frame {
public:
    enum class StorageType {
        EMPTY,
        OPENCV,          // cv::Mat
        VIDEO_FRAME      // VIDEO_FRAME_INFO_S (zero-copy)
    };

    // 构造
    Frame();
    explicit Frame(const cv::Mat& mat);
    explicit Frame(const VIDEO_FRAME_INFO_S& vf, bool owns);

    // 属性
    int width() const;
    int height() const;
    int channels() const;
    PixelFormat format() const;
    StorageType storage_type() const;
    bool empty() const;

    // Zero-copy支持
    bool has_physical_addr() const;
    uint64_t physical_addr() const;
    size_t physical_size() const;

    // 转换（lazy）
    const cv::Mat& to_mat() const;
    const VIDEO_FRAME_INFO_S& to_video_frame() const;

    // 资源管理
    void release();  // 释放VPSS/VI frame

private:
    std::variant<std::monostate, cv::Mat, VIDEO_FRAME_INFO_S> data_;
    StorageType type_;
    bool owns_memory_;

    // Lazy conversion cache
    mutable cv::Mat mat_cache_;
    mutable bool mat_cache_valid_;
};
```

### 4.2 CvCamera - 相机接口

```cpp
class CvCamera {
public:
    virtual ~CvCamera() = default;

    virtual bool open() = 0;
    virtual bool read(Frame& frame) = 0;
    virtual void release() = 0;

    // Properties
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual double fps() const = 0;
    virtual bool is_opened() const = 0;
};
```

### 4.3 CviCamera - VI硬件Camera

```cpp
class CviCamera : public CvCamera {
public:
    struct Config {
        int width = 1920;
        int height = 1080;
        int fps = 30;
        PixelFormat format = PixelFormat::NV21;
        int sensor_type = 0;  // 0=auto, 1=OV5647, 2=GC2053
    };

    explicit CviCamera(const Config& config = Config());
    ~CviCamera() override;

    bool open() override;
    bool read(Frame& frame) override;  // 返回VIDEO_FRAME_INFO_S
    void release() override;

    int width() const override { return config_.width; }
    int height() const override { return config_.height; }
    double fps() const override { return config_.fps; }
    bool is_opened() const override { return opened_; }

private:
    void init_vb_pool();
    void init_vi_module();
    void init_vpss_module();
    void bind_vi_vpss();
    void start_pipeline();
    void stop_pipeline();

    Config config_;
    bool opened_ = false;

    // VI资源
    VI_DEV vi_dev_ = 0;
    VI_PIPE vi_pipe_ = 0;
    VI_CHN vi_chn_ = 0;

    // VPSS资源
    VPSS_GRP vpss_grp_ = 0;
    VPSS_CHN vpss_chn_ = 0;

    // VB资源
    VB_POOL vb_pool_ = VB_INVALID_POOLID;
};
```

### 4.4 CvProcessor - 图像处理接口

```cpp
class CvProcessor {
public:
    virtual ~CvProcessor() = default;

    virtual void resize(Frame& frame, int width, int height) = 0;
    virtual void cvtColor(Frame& frame, ColorConversion code) = 0;
    virtual void crop(Frame& frame, int x, int y, int w, int h) = 0;
};
```

### 4.5 CviVpssProcessor - VPSS硬件处理器（已实现）

**核心职责**：
- 使用 VPSS 硬件加速图像处理操作
- 自动处理两种输入类型：
  1. VIDEO_FRAME_INFO_S（零拷贝，来自 Camera）
  2. cv::Mat（自动转换，来自文件）
- 管理 VB pool 资源分配和释放
- 实现 pipeline 缓存优化

**关键设计决策**：

1. **智能输入处理**
   - 检测 `frame.has_physical_addr()`
   - 有物理地址 → 直接零拷贝使用
   - 无物理地址 → 调用 `mat_to_video_frame()` 自动转换

2. **VB Pool 管理策略**
   - 使用独立 VB pool（与 Camera pipeline 分离）
   - 动态按需分配（CVI_VB_GetBlock）
   - 及时释放避免泄漏（CVI_VB_ReleaseBlock）
   - 64 字节对齐保证 DMA 性能

3. **Pipeline 缓存机制**
   - 缓存当前 VPSS 配置（input/output 尺寸、格式）
   - 参数不变时复用 pipeline（避免重新初始化）
   - 参数改变时重建 pipeline

4. **内存对齐要求**
   - Stride 必须 64 字节对齐：`ALIGN(width * bpp, 64)`
   - 物理地址 64 字节对齐（VB pool 保证）
   - 遵循 VPSS DMA 硬件限制

**mat_to_video_frame 设计**：

此功能负责将 CPU 内存（cv::Mat）转换为 VPSS 可处理的 VIDEO_FRAME_INFO_S。

转换流程：
1. 计算所需内存大小（考虑 stride 对齐）
2. 从 VB pool 分配物理内存块
3. 获取物理地址和虚拟地址映射
4. 拷贝数据（CPU → VB，支持 stride）
5. 取消虚拟地址映射
6. 构造 VIDEO_FRAME_INFO_S 结构
7. 记录 VB block 用于后续释放

内存生命周期：
- VB block 在 mat_to_video_frame() 中分配
- 所有权传递给 Frame（owns_memory=true）
- Frame 析构时自动释放 VB block
- VPSS 处理完成后立即可释放（无需等待 TPU）

格式支持：
- RGB888 / BGR888：3 字节/像素，stride = ALIGN(width*3, 64)
- NV12 / NV21：1.5 字节/像素，Y+UV 分离平面
- GRAY：1 字节/像素

**资源配置**：
- VPSS_GRP: 1（独立组，避免与 Camera 冲突）
- VPSS_CHN: 0
- VB_POOL: 动态分配（按需创建/销毁）

**性能特征**：
- 零拷贝输入：~8ms（纯 VPSS）
- 文件输入：~15ms（5-7ms 拷贝 + 8ms VPSS）
- Pipeline 初始化：<1ms（首次或参数变化时）
- Pipeline 复用：~0.01ms（参数未变时）

### 4.6 OpenCvProcessor - CPU处理器

```cpp
class OpenCvProcessor : public CvProcessor {
public:
    void resize(Frame& frame, int w, int h) override;
    void cvtColor(Frame& frame, ColorConversion code) override;
    void crop(Frame& frame, int x, int y, int w, int h) override;

    // 使用OpenCV实现，约25ms for 1080p→640x640
};
```

### 4.7 IonImageLoader - VB Pool统一图像加载器 ✅

**核心职责**：
- 统一图像加载（JPEG文件 → VIDEO_FRAME_INFO_S）
- 支持硬件JPEG解码（VDEC）和软件解码（OpenCV）
- 使用 VB pool 替代 Ion 分配（避免资源冲突）
- 提供预分配优化路径

**关键设计决策**：

1. **VB Pool 统一管理**
   - 问题：`CVI_SYS_IonAlloc_Cached` 与 VB pools 共享 Ion 后端，导致 "ioctl SYS_ION_ALLOC failed"
   - 解决：使用 `CVI_VB_GetBlock(VB_INVALID_POOLID, size)` 从公共 VB pool 分配
   - 优势：与 VPSS/VDEC 共享资源池，避免碎片化

2. **双解码器架构**
   - `decode_with_opencv()`: 软件解码（cv::imdecode, 134ms）
   - `decode_with_vdec()`: 硬件解码（HwJpegDecoder, 3.9ms）
   - 自动回退：VDEC 失败时 fallback 到 OpenCV

3. **内存生命周期**
   - VB block 在 `load()` 中分配
   - 通过 `VIDEO_FRAME_INFO_S` 返回给调用者
   - 调用者负责释放（RAII 或手动）
   - 支持 preallocate + load_from_memory_fast（避免重复分配）

**API 设计**：
```cpp
class IonImageLoader {
public:
    enum JpegDecoder {
        JPEG_DECODER_OPENCV,  // 软件解码（默认）
        JPEG_DECODER_VDEC     // 硬件解码（推荐）
    };

    // 加载JPEG文件
    VIDEO_FRAME_INFO_S load(const std::string& filepath);

    // 从内存加载
    VIDEO_FRAME_INFO_S load_from_memory(const uint8_t* data, size_t size);

    // 预分配优化
    bool preallocate(uint32_t width, uint32_t height, uint32_t channels);
    VIDEO_FRAME_INFO_S load_from_memory_fast(const uint8_t* data, size_t size);

    // 设置解码器类型
    void set_jpeg_decoder(JpegDecoder decoder);

private:
    VB_BLK vb_block_;          // VB block handle
    CVI_U64 vb_phys_addr_;     // Physical address
    void* vb_virt_addr_;       // Virtual address (cached)
    uint32_t vb_size_;         // Buffer size

    HwJpegDecoder* vdec_decoder_;  // VDEC解码器实例
    JpegDecoder jpeg_decoder_;     // 解码器选择
};
```

**性能数据（1280x720 JPEG）**：
- OpenCV 路径：141.8 ms（imread + VB memcpy）
- VDEC 路径：**3.9 ms**（硬件解码，36x 加速）
- 预分配路径：124.6 ms（复用 VB buffer）

### 4.8 HwJpegDecoder - VDEC硬件JPEG解码器 ✅

**核心职责**：
- 封装 SG200X VDEC 硬件 JPEG 解码
- 输出 VIDEO_FRAME_INFO_S（NV21 格式，物理地址）
- 支持同步和异步两种模式
- 零拷贝输出，直接对接 VPSS

**关键优化**：

1. **阻塞模式优化（decode_sync）**
   - 问题：异步线程池模式有 50ms 固定延迟（usleep 同步）
   - 解决：使用 `CVI_VDEC_SendStream(-1)` 和 `CVI_VDEC_GetFrame(-1)` 阻塞模式
   - 效果：**51.8ms → 3.9ms（13.3x 加速）**

2. **VDEC 配置**
   - Payload Type: PT_JPEG
   - Mode: VIDEO_MODE_FRAME（完整 JPEG 帧）
   - VB Source: VB_SOURCE_COMMON（使用公共 VB pool）
   - Pixel Format: PIXEL_FORMAT_YUV_PLANAR_444（JPEG 默认）
   - Output Format: NV21（自动转换）

3. **双模式支持**
   - `decode()`: 异步模式，使用永久线程池（兼容性）
   - `decode_sync()`: 同步模式，阻塞 API（推荐，高性能）
   - `decode_file()` / `decode_file_sync()`: 文件加载便捷接口

**API 设计**：
```cpp
class HwJpegDecoder {
public:
    // 初始化VDEC通道
    bool init(uint32_t max_width, uint32_t max_height);

    // 异步解码（线程池模式，51.8ms）
    VIDEO_FRAME_INFO_S decode(const uint8_t* data, size_t size);
    VIDEO_FRAME_INFO_S decode_file(const std::string& filepath);

    // 同步解码（阻塞模式，3.9ms）✅ 推荐
    VIDEO_FRAME_INFO_S decode_sync(const uint8_t* data, size_t size);
    VIDEO_FRAME_INFO_S decode_file_sync(const std::string& filepath);

    // 释放解码帧
    void release_frame(const VIDEO_FRAME_INFO_S& frame);

    void cleanup();

private:
    VDEC_CHN vdec_chn_;     // VDEC channel (0)
    bool initialized_;

    // 异步模式：永久线程池
    pthread_t send_thread_;
    pthread_t get_thread_;
    // ... 线程同步机制
};
```

**性能对比（1280x720 JPEG）**：
- 软件解码（cv::imdecode）: 134.25 ms
- 硬件异步（decode）: 51.75 ms（2.6x 加速，但有线程延迟）
- **硬件同步（decode_sync）**: **3.91 ms**（34.4x 加速）✅

**集成方式**：
- IonImageLoader 使用 `decode_sync()` 作为默认 VDEC 路径
- 输出直接对接 VPSS（零拷贝）
- 支持任意尺寸 JPEG（动态分配 VDEC buffer）

## 5. 智能路径选择策略

### 5.1 自动选择逻辑（cv_helpers）

```
输入 Frame
    │
    ├─ [USE_CVI_MPI 已启用]
    │   │
    │   ├─ has_physical_addr() == true
    │   │   → 路径 A：零拷贝 VPSS
    │   │
    │   └─ has_physical_addr() == false
    │       → 路径 B：mat_to_video_frame + VPSS
    │
    └─ [USE_CVI_MPI 未启用]
        → 路径 C：CPU OpenCV
```

### 5.2 容错机制

VPSS 路径失败时自动降级到 CPU：
- VB pool 资源耗尽
- VPSS 硬件初始化失败
- 不支持的像素格式
- 内存分配失败

降级策略：
- 输出警告日志
- 自动切换到 OpenCvProcessor
- 保证功能正确性（性能降级但不中断）

## 6. 资源初始化与清理

### 6.1 CviCamera初始化顺序

```cpp
1. CVI_VB_Init()              // VB系统初始化
2. CVI_SYS_Init()             // 系统初始化
3. CVI_VB_CreatePool()        // 创建VB pool
4. CVI_VI_SetDevAttr()        // 设置VI设备属性
5. CVI_VI_EnableDev()         // 使能VI设备
6. CVI_VI_CreatePipe()        // 创建VI pipe
7. CVI_VI_StartPipe()         // 启动VI pipe
8. CVI_VI_SetChnAttr()        // 设置VI channel属性
9. CVI_VI_EnableChn()         // 使能VI channel
10. CVI_VPSS_CreateGrp()      // 创建VPSS group
11. CVI_VPSS_ResetGrp()       // 重置VPSS group
12. CVI_VPSS_SetChnAttr()     // 设置VPSS channel属性
13. CVI_VPSS_AttachVbPool()   // 绑定VB pool到VPSS
14. CVI_VPSS_EnableChn()      // 使能VPSS channel
15. CVI_VPSS_StartGrp()       // 启动VPSS group
16. CVI_SYS_Bind()            // VI → VPSS绑定
```

### 6.2 CviCamera清理顺序

```cpp
1. CVI_SYS_UnBind()           // 解除VI → VPSS绑定
2. CVI_VPSS_StopGrp()         // 停止VPSS
3. CVI_VPSS_DisableChn()      // 禁用VPSS channel
4. CVI_VPSS_DetachVbPool()    // 解除VB pool绑定
5. CVI_VPSS_DestroyGrp()      // 销毁VPSS group
6. CVI_VI_DisableChn()        // 禁用VI channel
7. CVI_VI_StopPipe()          // 停止VI pipe
8. CVI_VI_DestroyPipe()       // 销毁VI pipe
9. CVI_VI_DisableDev()        // 禁用VI设备
10. CVI_VB_DestroyPool()      // 销毁VB pool
11. CVI_SYS_Exit()            // 系统退出
12. CVI_VB_Exit()             // VB系统退出
```

## 7. Lua API设计

### 7.1 Zero-Copy Pipeline（推荐用法）

```lua
-- Camera capture with zero-copy
local camera = cv.Camera({
    width = 1920,
    height = 1080,
    fps = 30,
    format = "NV21"
})

if not camera:open() then
    error("Failed to open camera")
end

while true do
    local frame = camera:read()  -- VIDEO_FRAME_INFO_S (zero-copy)
    if frame:empty() then break end

    -- VPSS hardware resize (~8ms)
    cv.resize(frame, 640, 640)

    -- Zero-copy TPU inference
    local input = nn.TpuTensor.from_frame(frame)
    local output = session:run(input)

    -- Process output...

    frame:release()  -- 释放VPSS frame
end

camera:release()
```

### 7.2 文件硬件加速 Pipeline

用户无需关心底层实现，cv_helpers 自动使用 VPSS 硬件加速：
- cv::imread() 读取到 CPU 内存
- 内部自动转换为 VIDEO_FRAME_INFO_S（拷贝到 VB pool）
- VPSS 硬件加速处理
- 自动释放临时 VB 资源

### 7.3 API列表

```lua
-- Camera
cv.Camera(config)              -- 创建Camera对象
camera:open()                  -- 打开Camera
camera:read()                  -- 读取一帧（返回Frame）
camera:release()               -- 释放Camera
camera:is_opened()             -- 检查是否已打开

-- Frame operations
cv.imread(path)                -- 读取图像
cv.imwrite(path, frame)        -- 保存图像
cv.resize(frame, width, height) -- 调整大小（自动选择CPU/VPSS）
cv.cvtColor(frame, code)       -- 颜色转换
cv.crop(frame, x, y, w, h)     -- 裁剪

-- Frame methods
frame:width()                  -- 获取宽度
frame:height()                 -- 获取高度
frame:channels()               -- 获取通道数
frame:empty()                  -- 检查是否为空
frame:has_physical_addr()      -- 是否有物理地址（zero-copy）
frame:release()                -- 释放资源（VPSS frame需要）
```

## 9. 内存管理

### 9.1 零拷贝路径（Camera 输入）

- Frame 持有 VIDEO_FRAME_INFO_S，owns_memory=false
- 内存由 VI/VPSS VB pool 管理
- Frame 析构时调用 CVI_VPSS_ReleaseChnFrame()
- VB pool 自动循环复用

**生命周期**：
```
Camera.read() → 从 VB pool 获取
    ↓
Frame 持有（owns=false）
    ↓
VPSS 处理（零拷贝）
    ↓
Frame.release() → 归还 VB pool
    ↓
VB pool 循环使用
```

### 9.2 文件硬件加速路径（新增）

- 临时 VB block 在 mat_to_video_frame() 中分配
- Frame 持有 VIDEO_FRAME_INFO_S，owns_memory=true
- Frame 析构时调用 CVI_VB_ReleaseBlock() 释放
- 及时释放避免 VB pool 耗尽

**生命周期**：
```
cv::imread() → CPU 内存
    ↓
mat_to_video_frame() → 分配 VB block + 拷贝数据
    ↓
Frame 持有（owns=true，带 VPSS context）
    ↓
VPSS 处理（硬件加速）
    ↓
Frame.release() → 释放 VB block
    ↓
VB pool 可用空间恢复
```

**内存分配策略**：
- 按需分配：CVI_VB_GetBlock(VB_INVALID_POOLID, size)
- 使用公共 VB pool（与 Camera 独立）
- 处理完成立即释放（不长期持有）
- 失败时自动 fallback 到 CPU 路径

**防止内存泄漏**：
- Frame 使用 RAII 模式（构造/析构自动管理）
- owns_memory_ 标志明确所有权
- vpss_grp_/vpss_chn_ 记录释放上下文
- 移动语义正确转移所有权

### 9.3 CPU 路径

- Frame 持有 cv::Mat，owns_memory=true
- 内存由 cv::Mat 自动管理（引用计数）
- 无需手动释放

### 9.4 VB Pool 配置

**Camera Pipeline**（CviCamera 使用）：
- Pool size: 8MB × 3 blocks = 24MB
- 用途：VI → VPSS 循环缓冲
- 长期占用（Camera 生命周期）

**Processor Pool**（CviVpssProcessor 使用）：
- 动态分配（按需）
- 用途：文件输入临时转换
- 短期占用（处理完成即释放）

**总内存预算**（最坏情况）：
- Camera pool: 24MB（如启用）
- 临时 VB block: ~6MB（单帧，1080p RGB）
- OpenCV Mat: ~6MB（单帧）
- 总计：<40MB（可接受）

### 9.5 资源泄漏检测

**调试模式**：
- 记录所有 VB block 分配/释放
- 统计 outstanding VB blocks
- 超过阈值时输出警告

**生产模式**：
- 轻量级检测（仅计数）
- 异常时降级到 CPU 路径
- 保证程序不崩溃

## 10. 错误处理

### 硬件初始化失败
```cpp
try {
    camera.open();  // 尝试硬件
} catch (const std::exception& e) {
    // 自动fallback到CPU（如果可用）
    // 或抛出异常告知用户
}
```

### VB Pool耗尽
```cpp
// GetChnFrame with timeout
CVI_VPSS_GetChnFrame(grp, chn, &frame, 1000);  // 1秒超时
// 超时返回错误码，不会无限等待
```

### 资源泄漏检测
```cpp
// Debug模式记录所有未释放的frame
#ifdef DEBUG
std::atomic<int> outstanding_frames{0};
#endif
```

## 11. 测试计划

### 11.1 单元测试
- [x] Frame构造/转换正确性
- [ ] CviCamera帧获取（30fps稳定）
- [ ] VPSS resize精度（误差<1%）
- [ ] CPU fallback功能
- [ ] 内存泄漏检测（1000帧）

### 11.2 性能测试
- [ ] Camera → VPSS resize: **< 10ms** ✓
- [ ] CPU resize: **< 30ms** ✓
- [ ] 完整pipeline (Camera → TPU): **< 40ms** ✓
- [ ] FPS测试: **> 25 FPS** ✓

### 11.3 压力测试
- [ ] 连续运行24小时无崩溃
- [ ] 内存占用稳定（无泄漏）
- [ ] CPU/内存使用率合理

### 11.4 Lua集成测试
- [ ] 所有API可从Lua调用
- [ ] 错误处理正确
- [ ] 示例脚本可运行

## 12. 实现状态

### Phase 1: 核心框架 ✅ 完成
- [x] cv_design.md（包含 mat_to_video_frame 设计）
- [x] cv_types.h（颜色转换枚举等）
- [x] cvi_frame.h/cpp（支持双存储类型 + VPSS context）

**实现细节**：
- Frame 支持 cv::Mat 和 VIDEO_FRAME_INFO_S
- 添加 vpss_grp_/vpss_chn_ 用于内存管理
- 实现 release_video_frame() 正确释放资源

### Phase 2: 硬件 Camera ⏸️ 暂停
- [x] cvi_camera.h/cpp（代码完成）
- [ ] ISP 库依赖解决（阻塞中）

**状态**：代码已实现但未构建，等待 ISP 库链接问题解决。

### Phase 3: 硬件处理器 ✅ 完成
- [x] cvi_vpss_processor.h/cpp
- [x] resize() 实现（支持零拷贝和文件输入）
- [x] cvtColor() 实现（格式转换）
- [x] crop() 实现（硬件裁剪）
- [x] mat_to_video_frame() 实现（文件输入硬件加速）

### Phase 4: CPU Fallback ✅ 完成
- [x] opencv_processor.h/cpp（所有操作）
- [x] resize, cvtColor, crop 全部实现

### Phase 4.5: 智能后端选择 ✅ 完成（新增阶段）
- [x] cv_helpers.h/cpp（统一 API）
- [x] 自动路径选择逻辑
- [x] 容错机制（VPSS 失败 fallback CPU）

**实现亮点**：
- 用户无需关心后端细节
- 自动优化性能
- 降级机制保证可靠性

### Phase 4.6: 文件硬件加速全链路优化 ✅ 完成（新增阶段）

#### IonImageLoader（VB Pool 统一管理）✅
- [x] ion_image_loader.h/cpp
- [x] 从 Ion 分配迁移到 VB pool（`CVI_VB_GetBlock`）
- [x] 双解码器架构（OpenCV + VDEC）
- [x] 预分配优化路径（`preallocate` + `load_from_memory_fast`）
- [x] JPEG 格式自动检测（magic number）
- [x] JPEG 头解析（获取尺寸）

**解决问题**：
- ✅ 修复 "ioctl SYS_ION_ALLOC failed" 错误
- ✅ 统一 VB pool 资源管理
- ✅ 避免内存碎片化

**性能数据（1280x720 JPEG）**：
- OpenCV 路径：141.8 ms
- VDEC 路径：3.9 ms（36x 加速）

#### HwJpegDecoder（VDEC 硬件加速）✅
- [x] hw_jpeg_decoder.h/cpp
- [x] VDEC 通道初始化（PT_JPEG 模式）
- [x] 异步解码（`decode`, 线程池模式）
- [x] **同步解码（`decode_sync`, 阻塞模式）** ← 关键优化
- [x] 文件加载接口（`decode_file` / `decode_file_sync`）
- [x] 永久线程池（避免频繁创建/销毁）
- [x] Cache invalidation（`CVI_SYS_IonInvalidateCache`）

**关键优化**：
- ✅ 阻塞模式消除 50ms 线程同步延迟
- ✅ 使用 VB_SOURCE_COMMON（公共 VB pool）
- ✅ 输出 NV21 格式（直接对接 VPSS）

**性能对比（1280x720 JPEG）**：
- 软件解码：134.25 ms
- 硬件异步：51.75 ms（2.6x 加速）
- **硬件同步：3.91 ms（34.4x 加速）** ✅

#### 全链路集成测试 ✅
- [x] test_ion_loader.cpp（综合性能测试）
- [x] 对比 OpenCV / VDEC async / VDEC sync
- [x] 内存分配性能测试
- [x] VB pool 管理测试
- [x] 全流程 benchmark（file read + decode + VPSS）

**测试结果（1280x720 JPEG → 640x640）**：
- CPU 全流程：145.2 ms
- VDEC sync + VPSS：**12 ms**（12x 加速）✅
- 所有 58 项测试通过 ✅

### Phase 5: Lua 绑定与 Image/Frame 统一 ✅ 设计完成，待实现

**设计目标**：
- 统一 Image 和 Frame，对 Lua 透明地提供硬件加速
- 保持现有 Lua API 不变，向后兼容
- cv_helpers 不暴露给 Lua，保持抽象层次清晰
- 为未来 Pipeline 架构预留接口

#### 5.1 统一架构设计

**核心思路**：Image 类内部使用 Frame 实现，而非直接使用 cv::Mat

**分层关系**：
```
Lua 层（用户接口）
  ↓
Image 类（lua_cv 模块，API facade）
  ↓
cv_helpers（智能后端选择，不暴露）
  ↓
Frame 类（cv 模块，双存储后端）
  ↓
VPSS 硬件 / OpenCV CPU
```

**关键设计决策**：

1. **Image 作为 Facade**
   - Lua 仅看到 Image 类，不直接操作 Frame
   - Image 内部持有 Frame 对象（组合模式）
   - 所有操作委托给 cv_helpers，由其选择最优路径

2. **cv_helpers 作为抽象层**
   - 不暴露给 Lua，仅 C++ 内部使用
   - 负责智能后端选择（VPSS vs OpenCV）
   - 处理错误和 fallback 逻辑
   - 提供 frame_to_tensor 转换

3. **Frame 保持内部实现**
   - 支持双存储类型（cv::Mat / VIDEO_FRAME_INFO_S）
   - 提供零拷贝能力（物理地址模式）
   - Lua 不直接访问，通过 Image 间接使用

#### 5.2 数据流分析

**Camera 输入零拷贝路径**：
```
CviCamera::read()
  → Frame(VIDEO_FRAME_INFO_S, 物理地址)
  → Image(frame_)
  → Lua: img:resize(640,640)
  → cv_helpers::resize(frame_)
  → 检测 has_physical_addr() == true
  → VPSS 硬件零拷贝处理 (~8ms)
  → img:to_tensor()
  → cv_helpers::frame_to_tensor()
  → 映射物理地址，HWC→CHW 转换 (~5ms)
  → Tensor
```

**文件输入硬件加速路径**：
```
cv::imread()
  → cv::Mat
  → Frame(mat)
  → Image(frame_)
  → Lua: img:resize(640,640)
  → cv_helpers::resize(frame_)
  → 检测 has_physical_addr() == false
  → mat_to_video_frame 转换 + VB pool 拷贝 (~5-7ms)
  → VPSS 硬件处理 (~8ms)
  → img:to_tensor()
  → cv_helpers::frame_to_tensor()
  → HWC→CHW 转换 (~5ms)
  → Tensor
```

**CPU Fallback 路径**：
```
cv::imread()
  → cv::Mat
  → Frame(mat)
  → Image(frame_)
  → Lua: img:resize(640,640)
  → cv_helpers::resize(frame_)
  → 检测 USE_CVI_MPI 未启用或 VPSS 失败
  → OpenCV CPU 处理 (~75ms)
  → img:to_tensor()
  → 直接从 cv::Mat 转换
  → Tensor
```

#### 5.3 关键实现要点

**Image 类重构**：
- 将内部 `cv::Mat mat_` 替换为 `lua_cv::Frame frame_`
- 所有属性访问（width/height/channels）委托给 Frame
- 所有图像操作（resize/pad）通过 cv_helpers 调用
- to_tensor() 方法调用 cv_helpers::frame_to_tensor()

**imread 全局函数**：
- 使用 cv::imread 加载图像到 cv::Mat
- BGR→RGB 转换（OpenCV 默认 BGR）
- 创建 Frame 包装 cv::Mat
- 返回 Image 对象

**cv_helpers 扩展**：
- 新增 frame_to_tensor() 函数
- 接受 Frame 引用和归一化参数
- 内部调用 frame.to_mat() 获取数据
- 执行 HWC→CHW 转换和归一化
- 返回 NCHW 格式的 Tensor

**pad 操作特殊处理**：
- VPSS 不支持 pad 操作（硬件限制）
- 必须使用 OpenCV CPU 实现
- 需要确保 Frame 转换为 cv::Mat
- 操作后重新包装为 Frame

#### 5.4 向后兼容性

**Lua API 完全不变**：
- 所有现有脚本无需修改
- imread/resize/pad/to_tensor 签名不变
- 属性访问方式不变
- 错误处理行为一致

**自动性能提升**：
- Camera 输入自动获得零拷贝优化
- 文件输入自动获得硬件加速
- 用户代码无感知，透明升级

**共存策略**：
- 保持 lua_cv 模块名称不变
- Image 类名不变
- 仅内部实现从 cv::Mat 切换到 Frame

#### 5.5 向 Pipeline 架构过渡

**未来 Pipeline 集成点**：

1. **Context 传递**
   - C++ Pipeline 创建 Frame 对象（Camera 或 File）
   - 包装为 Image 对象传递给 Lua
   - Lua preprocess 函数接收 ctx.image（Image 类型）

2. **零拷贝保持**
   - Frame 携带物理地址信息
   - Image 不破坏零拷贝特性
   - 整个链路保持高性能

3. **抽象层隔离**
   - Lua 只操作 Image（高层抽象）
   - Pipeline 内部操作 Frame（底层实现）
   - cv_helpers 作为桥接层

4. **灵活扩展**
   - 未来可添加 Image 新方法而不改变 Frame
   - Pipeline 可直接操作 Frame 优化性能
   - 两层解耦，各自演进

#### 5.6 性能预期

**Camera 零拷贝路径**：
- 总预处理时间：~13ms
  - VPSS resize: ~8ms
  - to_tensor 转换: ~5ms
- 相比纯 CPU（~90ms）：**6.9x 加速**

**文件硬件加速路径**：
- 总预处理时间：~22ms
  - mat_to_video_frame: ~5-7ms
  - VPSS resize: ~8ms
  - to_tensor 转换: ~5ms
- 相比纯 CPU（~90ms）：**4.1x 加速**

**内存开销**：
- Image 对象：仅增加 Frame 引用（~8 字节指针）
- 无额外拷贝：Frame 共享底层存储
- VB pool 临时分配：仅文件路径使用，处理后立即释放

#### 5.7 实现顺序

1. **cv_helpers 扩展**
   - 添加 frame_to_tensor() 函数
   - 实现归一化和 HWC→CHW 转换

2. **Image 类重构**
   - 替换内部存储为 Frame
   - 重新实现所有方法（委托模式）
   - 保持 Lua 绑定不变

3. **imread 函数更新**
   - 返回基于 Frame 的 Image
   - 保持 BGR→RGB 转换

4. **测试验证**
   - 所有现有测试必须通过
   - 验证性能提升
   - 确认 API 兼容性

#### 5.8 Lua 绑定原则

**遵守项目规范**：
- 不直接使用 Lua C API
- 通过 LuaIntf 封装所有接口
- 类型安全，RAII 资源管理

**保持简洁性**：
- 隐藏 Frame、cv_helpers 等内部细节
- 仅暴露 Image 高层抽象
- 错误信息友好，不泄漏实现细节

**向后兼容**：
- 现有 lua_cv.Image API 不变
- 新功能通过可选参数添加
- 废弃功能保留兼容期

### Phase 6: 测试与优化 🔄 进行中
- [x] 单元测试（test_cv_func.cpp）
- [x] Frame 基本功能测试
- [x] OpenCvProcessor 测试
- [x] CviVpssProcessor 错误处理测试
- [x] cv_helpers 智能选择测试
- [x] 性能基准测试（CPU 路径）
- [ ] VPSS 硬件性能测试（需 Camera 或 mat_to_video_frame）
- [ ] 内存泄漏检测
- [ ] 压力测试（24小时）

**测试结果**（已完成）：
- 17/20 功能测试通过
- CPU 性能符合预期（127ms，RISC-V 限制）
- VPSS 逻辑正确（等待硬件测试验证）

### 实施优先级（更新后）

**高优先级**（性能关键）：
1. ✅ VPSS cvtColor/crop 实现
2. ✅ cv_helpers 智能选择
3. ✅ Frame 内存管理修复
4. ✅ mat_to_video_frame 实现
5. 🔄 **Phase 5: Image/Frame 统一与 Lua 绑定**（当前任务）

**中优先级**（功能完善）：
6. 📋 Camera ISP 库链接
7. 📋 VPSS 硬件性能验证
8. 📋 端到端测试（Camera → TPU）

**低优先级**（优化改进）：
9. 📋 VB pool 动态调整
10. 📋 多线程支持
11. 📋 性能监控 API

## 13. 参考资料

- **官方文档**: `/home/baozhu/storage/reCamera-OS/MediaProcessingSoftwareDevelopmentReference_en.pdf`
- **VI API**: `$SDK/cvi_mpi/include/cvi_vi.h`
- **VPSS API**: `$SDK/cvi_mpi/include/cvi_vpss.h`
- **VB API**: `$SDK/cvi_mpi/include/cvi_vb.h`
- **参考实现**: `$SDK/cvi_mpi/sample/`

---

### Phase 7: 完整 Camera 支持（VI → ISP → VPSS）📋 设计完成

**目标**：实现完整的 Camera 到 AI 推理零拷贝流水线，支持 OV5647/GC2053 双传感器自适应。

#### 7.1 资源分配策略（更新）

| 模块 | VPSS_GRP | VPSS_CHN | VDEC_CHN | VI_DEV | VI_PIPE | VI_CHN | 备注 |
|------|----------|----------|----------|--------|---------|--------|------|
| CviCamera | 0 | 0 | - | 0 | 0 | 0 | Camera专用，VI绑定模式 |
| CviVpssProcessor | 1 | 0 | - | - | - | - | 图像处理，SendFrame模式 |
| HwJpegDecoder | - | - | 0 | - | - | - | JPEG硬解码 |

**数据流**：
```
路径1: Camera零拷贝 (全程物理地址)
Sensor → MIPI → VI(DEV0) → ISP → VPSS(GRP0) → TPU

路径2: JPEG硬件加速 (VDEC→VPSS)
JPEG File → VDEC(CHN0) → VIDEO_FRAME → SendFrame → VPSS(GRP1) → TPU

路径3: 图片硬件加速 (VB Pool→VPSS)
Image File → imread → Mat → VB Pool → SendFrame → VPSS(GRP1) → TPU
```

#### 7.2 传感器驱动抽象层（cvi_sensor.h/cpp）

**支持传感器**：
- OV5647：5MP，1920x1080@30fps，I2C地址0x36
- GC2053：2MP，1920x1080@30fps，I2C地址0x3f

**自动检测机制**：
1. 设置 I2C 总线 (bus_id=2)
2. 配置 I2C 地址
3. 调用 `pfnSnsProbe()` 探测
4. 按优先级尝试：OV5647 → GC2053
5. 探测成功则使用该传感器

**初始化流程**（参考 sscma-example）：
```
CVI_SYS_VI_Open()
    ↓
Sensor_Start()
    ├── pfnSetBusInfo() - I2C总线配置
    ├── pfnPatchI2cAddr() - I2C地址
    ├── pfnRegisterCallback() - AE/AWB回调注册
    ├── pfnExpSensorCb() - 曝光回调
    │   ├── pfn_cmos_sensor_global_init()
    │   ├── pfn_cmos_set_image_mode()
    │   └── pfn_cmos_set_wdr_mode()
    └── pfnPatchRxAttr() - MIPI属性
    ↓
Mipi_Start()
    ├── CVI_MIPI_SetSensorReset(1) - 复位传感器
    ├── CVI_MIPI_SetMipiReset(1) - 复位MIPI
    ├── CVI_MIPI_SetMipiAttr() - 设置MIPI属性
    ├── CVI_MIPI_SetSensorClock(1) - 使能时钟
    └── CVI_MIPI_SetSensorReset(0) - 解除复位
    ↓
Dev_Start()
    ├── CVI_VI_SetDevAttr()
    └── CVI_VI_EnableDev()
    ↓
Pipe_Start()
    ├── CVI_VI_CreatePipe()
    └── CVI_VI_StartPipe()
    ↓
ISP_Init()
    ├── CVI_AE_Register() - AE算法注册
    ├── CVI_AWB_Register() - AWB算法注册
    ├── CVI_ISP_SetBindAttr() - 绑定AE/AWB
    ├── CVI_ISP_MemInit() - ISP内存初始化
    ├── CVI_ISP_SetPubAttr() - 公共属性
    └── CVI_ISP_Init()
    ↓
ISP_Start()
    └── pthread_create(ISP_Thread) - ISP运行线程
        └── CVI_ISP_Run(ViPipe) - ISP主循环
    ↓
Chn_Start()
    ├── CVI_VI_SetChnAttr()
    ├── CVI_VI_RegChnFlipMirrorCallBack() - 镜像翻转回调
    └── CVI_VI_EnableChn()
```

**API设计**：
```cpp
class CviSensor {
public:
    enum class SensorType { NONE, OV5647, GC2053, AUTO };

    struct Config {
        SensorType type = SensorType::AUTO;  // 自动检测
        int32_t bus_id = 2;                  // I2C总线
        uint32_t width = 1920;
        uint32_t height = 1080;
        int32_t framerate = 30;
        bool mirror = false;
        bool flip = false;
    };

    bool init(const Config& config);  // 初始化（包含ISP）
    void cleanup();                   // 清理资源

    SensorType get_sensor_type() const;  // 获取检测到的传感器类型
    const char* get_sensor_name() const; // 获取传感器名称
    VI_PIPE get_vi_pipe() const;         // 获取VI管道ID
    VI_CHN get_vi_chn() const;           // 获取VI通道ID
};
```

#### 7.3 CviCamera 更新（集成 CviSensor）

**当前问题**：
- 缺少 ISP 初始化（AE/AWB）
- 缺少传感器驱动初始化
- 缺少 MIPI 配置

**更新方案**：
```cpp
class CviCamera {
public:
    bool open() {
        // 1. 使用 CviSensor 初始化传感器 + ISP
        if (!sensor_.init(sensor_config_)) {
            return false;
        }

        // 2. 初始化 VPSS（GRP=0，与 VI 绑定）
        init_vpss_module();

        // 3. 绑定 VI → VPSS
        bind_vi_vpss();

        return true;
    }

    bool read(Frame& frame) {
        // 从 VPSS 获取帧（零拷贝）
        VIDEO_FRAME_INFO_S vpss_frame;
        CVI_VPSS_GetChnFrame(vpss_grp_, vpss_chn_, &vpss_frame, timeout);
        frame = Frame(vpss_frame, vpss_grp_, vpss_chn_);
        return true;
    }

private:
    CviSensor sensor_;  // 传感器管理
    VPSS_GRP vpss_grp_ = 0;  // Camera专用VPSS组
    VPSS_CHN vpss_chn_ = 0;
};
```

#### 7.4 链接依赖

**传感器驱动库**（需在 CMakeLists.txt 添加）：
- `libsns_ov5647.a` - OV5647传感器驱动
- `libsns_gc2053.a` - GC2053传感器驱动

**ISP库**：
- `libcvi_bin.so` - ISP参数二进制加载
- `libae.so` - 自动曝光
- `libawb.so` - 自动白平衡
- `libisp.so` - ISP核心

**头文件路径**：
```
$SG200X_SDK_PATH/cvi_mpi/include/
  ├── cvi_ae.h
  ├── cvi_awb.h
  ├── cvi_isp.h
  ├── cvi_mipi.h
  └── cvi_bin.h
```

#### 7.5 测试计划

**测试文件**：
1. `test_camera.cpp` - Camera → VI → ISP → VPSS 测试
2. `test_jpeg_vpss.cpp` - JPEG → VDEC → VPSS 测试（验证全链路）

**测试矩阵**：
| 测试用例 | 输入源 | 管道 | 验证项 |
|---------|--------|------|--------|
| Camera基本 | 传感器 | VI → VPSS(GRP0) | 获取帧成功 |
| Camera+ISP | 传感器 | VI → ISP → VPSS | AE/AWB工作 |
| JPEG→VDEC | JPEG文件 | VDEC → VIDEO_FRAME | 硬解码成功 |
| JPEG→VPSS | JPEG文件 | VDEC → SendFrame → VPSS(GRP1) | resize成功 |
| 图片→VPSS | PNG文件 | imread → VB → VPSS(GRP1) | resize成功 |
| 并发测试 | Camera+JPEG | 同时运行两条流水线 | 无资源冲突 |

**设备测试命令**：
```bash
# Camera测试
sshpass -p '11' ssh recamera@192.168.42.1 'echo "11" | sudo -S /tmp/test_camera'

# JPEG→VPSS测试
sshpass -p '11' ssh recamera@192.168.42.1 '/tmp/test_jpeg_vpss /tmp/test.jpg'
```

#### 7.6 实现步骤

1. **Step 1**: 创建 cvi_sensor.h/cpp（传感器抽象层）✅ 已完成文件框架
2. **Step 2**: 更新 CMakeLists.txt（添加传感器库链接）
3. **Step 3**: 更新 cvi_camera.cpp（集成 CviSensor）
4. **Step 4**: 创建 test_camera.cpp（Camera测试）
5. **Step 5**: 创建 test_jpeg_vpss.cpp（JPEG→VPSS测试）
6. **Step 6**: 设备测试验证

---

**最后更新**: 2026-01-16
**状态**: Phase 1-4.6 完成，Phase 5 设计完成待实现，Phase 7（Camera支持）设计完成
**下一步**: 实现 Phase 7（Camera完整支持）

**重要里程碑**：
- ✅ VPSS 硬件加速（Phase 3）: 6.8x 性能提升
- ✅ VB Pool 统一管理（Phase 4.6）: 解决资源冲突
- ✅ VDEC 硬件 JPEG 解码（Phase 4.6）: 34.4x 性能提升
- ✅ **全链路硬件加速（Phase 4.6）**: JPEG → VDEC → VPSS，12x 端到端加速
- 📋 **Camera完整支持（Phase 7）**: VI → ISP → VPSS 零拷贝流水线

