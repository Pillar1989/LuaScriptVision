#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <string>
#include <cmath>
#include <functional>

#include "device_type.h"
#include "device_buffer.h"
#include "sync_handle.h"

// 前向声明 Lua 相关类型
struct lua_State;
namespace LuaIntf {
    class LuaRef;
    template<typename T> class TensorView;
}

namespace tensor {

/**
 * Tensor - 多维张量类
 *
 * 特性：
 * - 多设备支持 (CPU, NPU, TPU) 通过 DeviceBuffer 抽象
 * - 零拷贝视图操作 (slice, transpose) 使用 stride-based 索引
 * - 高性能计算 (contiguous 内存优化，SIMD 友好)
 * - Lua 互操作 (通过 LuaIntf 绑定)
 *
 * 设计原则：
 * - 嵌入式友好：最小化内存分配，支持原地操作
 * - 性能优先：热路径内联，缓存设备类型
 * - 模块化实现：10个 .cpp 文件按功能分类
 */
class Tensor {
public:
    // ==================== 构造与工厂方法 ====================
    // 实现文件: tensor_core.cpp

    /// 从 std::vector 拷贝构造
    Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape);

    /// 从 std::vector 移动构造
    Tensor(std::vector<float>&& data, const std::vector<int64_t>& shape);

    /// 从原始指针构造（零拷贝或拷贝）
    /// @param owner 如果提供，共享所有权；否则拷贝数据
    Tensor(const float* data, const std::vector<int64_t>& shape,
           std::shared_ptr<DeviceBuffer> owner = nullptr);

    /// 内部构造（支持 strides 和 offset，用于零拷贝视图）
    Tensor(std::shared_ptr<DeviceBuffer> buffer,
           const std::vector<int64_t>& shape,
           const std::vector<int64_t>& strides,
           int64_t offset,
           bool contiguous);

    /// Lua 工厂方法 - C++ 数据
    static Tensor create(const std::vector<float>& data, const std::vector<int64_t>& shape);

    /// Lua 工厂方法 - 从 Lua table 解析
    static Tensor from_lua(lua_State* L, const LuaIntf::LuaRef& data_table,
                          const LuaIntf::LuaRef& shape_table);

    /// 静态工厂方法 - 生成等差数列张量
    /// @param start 起始值
    /// @param end 结束值 (不包含)
    /// @param step 步长 (默认 1.0)
    /// @return 1D CPU 张量
    static Tensor arange(float start, float end, float step = 1.0f);


    // ==================== 属性访问 ====================
    // 实现文件: tensor_core.cpp

    /// 获取形状 (shape)
    std::vector<int64_t> shape() const { return shape_; }

    /// 获取步长 (strides)
    std::vector<int64_t> strides() const { return strides_; }

    /// 获取维度数
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }

    /// 获取总元素数
    int64_t size() const;

    /// 获取指定维度的大小
    int64_t size(int dim) const;

    /// 检查是否连续存储
    bool is_contiguous() const { return contiguous_; }

    /// 获取设备类型（缓存，避免虚函数调用 - OPT-4）
    DeviceType device() const { return device_cache_; }

    /// 获取底层缓冲区对象
    std::shared_ptr<DeviceBuffer> buffer() const { return buffer_; }


    // ==================== Lua 互操作 ====================
    // 实现文件: tensor_core.cpp

    /// Lua 包装 - 单索引读取 (支持非连续)
    float get_lua(int idx) const;

    /// Lua 包装 - 单索引写入 (支持非连续)
    void set_lua(int idx, float value);

    /// Lua 包装 - 2D 元素访问
    float at2d(int64_t i, int64_t j) const { return at(i, j); }

    /// 提取列返回 Lua table (零拷贝，用于 YOLO - OPT-3)
    /// @return {{col1_data}, {col2_data}, ...} 行格式
    LuaIntf::LuaRef extract_columns_lua(lua_State* L, const std::vector<int64_t>& cols) const;

    // ==================== 元素访问（支持 stride）====================
    // 实现文件: tensor_core.cpp

    /// 1D 索引访问 (const)
    float at(int64_t i) const;

    /// 2D 索引访问 (const)
    float at(int64_t i, int64_t j) const;

    /// 多维索引访问 (const)
    float at(const std::vector<int64_t>& indices) const;

    /// 1D 索引访问 (non-const)
    float& at(int64_t i);

    /// 2D 索引访问 (non-const)
    float& at(int64_t i, int64_t j);


    // ==================== 数据指针访问 ====================
    // 实现文件: tensor_core.cpp

    /// 获取原始数据指针 (不含 offset)
    const float* raw_data() const;
    float* raw_data();

    /// 获取数据指针 (含 offset)
    const float* data() const;
    float* data();


    // ==================== 设备操作 ====================
    // 实现文件: tensor_device.cpp

    /// 迁移到指定设备 (同步)
    Tensor to(DeviceType device) const;

    /// 确保连续存储 (如已连续则返回自身)
    Tensor contiguous() const;

    /// 异步迁移到指定设备
    Tensor to_async(DeviceType device, SyncHandle* handle = nullptr) const;

    /// 同步等待异步操作完成
    void sync(SyncHandle* handle = nullptr) const;

    /// 检查数据是否就绪 (非阻塞)
    bool is_ready(SyncHandle* handle = nullptr) const;

    /// 零拷贝视图 (向后兼容)
    LuaIntf::TensorView<float> view();


    // ==================== 形状操作（零拷贝）====================
    // 实现文件: tensor_shape.cpp

    /// 切片操作 (零拷贝视图)
    Tensor slice(int dim, int64_t start, int64_t end, int64_t step = 1) const;

    /// 选择单个维度 (降维)
    Tensor select_dim(int dim, int64_t index) const;

    /// 获取列 (2D 张量专用)
    Tensor get_column(int64_t col_idx) const;

    /// 切片列 (2D 张量专用)
    Tensor slice_columns(int64_t start, int64_t end) const;

    /// 重塑形状 (需要连续)
    Tensor reshape(const std::vector<int64_t>& new_shape) const;

    /// 转置 (指定维度顺序)
    Tensor transpose(const std::vector<int>& dims) const;

    /// 转置 (反转所有维度)
    Tensor transpose() const;

    /// 压缩维度 (移除 size=1 的维度)
    Tensor squeeze(int dim = -1) const;

    /// 扩展维度 (插入 size=1 的维度)
    Tensor unsqueeze(int dim) const;


    // ==================== 数学运算 ====================
    // 实现文件: tensor_math.cpp

    /// 加法 (tensor + tensor)
    Tensor add(const Tensor& other) const;
    Tensor add(float scalar) const;

    /// 减法 (tensor - tensor/scalar)
    Tensor sub(const Tensor& other) const;
    Tensor sub(float scalar) const;

    /// 乘法 (tensor * tensor/scalar)
    Tensor mul(const Tensor& other) const;
    Tensor mul(float scalar) const;

    /// 除法 (tensor / tensor/scalar)
    Tensor div(const Tensor& other) const;
    Tensor div(float scalar) const;

    /// In-place 操作（避免内存分配 - OPT-5）
    Tensor& add_(float scalar);
    Tensor& sub_(float scalar);
    Tensor& mul_(float scalar);
    Tensor& div_(float scalar);

    /// 值域裁剪
    /// @param min_val 最小值
    /// @param max_val 最大值
    /// @return 裁剪后的张量
    Tensor clamp(float min_val, float max_val) const;


    // ==================== 激活函数 ====================
    // 实现文件: tensor_activation.cpp

    /// Sigmoid 激活
    Tensor sigmoid() const;

    /// Softmax (仅支持最后一维)
    Tensor softmax(int axis = -1) const;

    /// 指数函数
    Tensor exp_() const;

    /// 对数函数
    Tensor log_() const;


    // ==================== 比较操作 ====================
    // 实现文件: tensor_compare.cpp

    /// 大于
    Tensor gt(float threshold) const;

    /// 小于
    Tensor lt(float threshold) const;

    /// 大于等于
    Tensor ge(float threshold) const;

    /// 小于等于
    Tensor le(float threshold) const;

    /// 等于
    Tensor eq(float threshold) const;


    // ==================== 归约操作 ====================
    // 实现文件: tensor_reduction.cpp

    /// 求和
    Tensor sum(int axis = -1, bool keepdims = false) const;

    /// 平均值
    Tensor mean(int axis = -1, bool keepdims = false) const;

    /// 最大值
    Tensor max(int axis = -1, bool keepdims = false) const;

    /// 最小值
    Tensor min(int axis = -1, bool keepdims = false) const;

    /// Argmax (返回 Lua table)
    LuaIntf::LuaRef argmax_lua(lua_State* L, int axis = -1) const;

    /// Argmin (返回 Lua table)
    LuaIntf::LuaRef argmin_lua(lua_State* L, int axis = -1) const;

    /// 融合 max + argmax (单次遍历 - OPT-7)
    /// @return Lua table: {values = Tensor, indices = table}
    LuaIntf::LuaRef max_with_argmax(lua_State* L, int axis = -1) const;

    /// 获取单个元素
    float get_item(const std::vector<int64_t>& indices) const;

    /// 设置单个元素
    void set_item(const std::vector<int64_t>& indices, float value);

    /// 调试输出
    std::string to_string(int max_elements = 10) const;


    // ==================== 选择与索引 ====================
    // 实现文件: tensor_select.cpp

    /// 返回非零元素的索引
    std::vector<int64_t> nonzero() const;

    /// 条件过滤索引 (用于 YOLO)
    /// @param op "ge"|"gt"|"le"|"lt"|"eq"
    std::vector<int64_t> where_indices(float threshold, const std::string& op = "ge") const;

    /// 根据索引选择元素
    Tensor index_select(int dim, const std::vector<int64_t>& indices) const;

    /// Top-K 选择 (优化实现 O(n + k log k))
    LuaIntf::LuaRef topk_lua(lua_State* L, int k, int axis = -1, bool largest = true) const;

    /// 转换为 Lua table
    LuaIntf::LuaRef to_table(lua_State* L) const;


    // ==================== 高级操作 ====================
    // 实现文件: tensor_advanced.cpp

    /// Gather 操作
    Tensor gather(int axis, const Tensor& indices) const;

    /// 拼接张量
    static Tensor concat(const std::vector<Tensor>& tensors, int axis);

    /// 分割张量
    std::vector<Tensor> split(int num_splits, int axis) const;


    // ==================== 遗留方法（向后兼容）====================
    // 实现文件: tensor_legacy.cpp

    /// Legacy YOLO 检测器过滤
    LuaIntf::LuaRef filter_yolo(lua_State* L, float conf_thres);

    /// Legacy YOLO 姿态估计过滤
    LuaIntf::LuaRef filter_yolo_pose(lua_State* L, float conf_thres);

    /// Legacy YOLO 分割过滤
    LuaIntf::LuaRef filter_yolo_seg(lua_State* L, float conf_thres);

    /// Legacy Mask 处理
    LuaIntf::LuaRef process_mask(lua_State* L, const LuaIntf::LuaRef& mask_coeffs,
                                  const LuaIntf::LuaRef& box,
                                  int img_w, int img_h,
                                  int input_w, int input_h,
                                  int pad_x, int pad_y);

    /// Legacy 分类 Argmax
    LuaIntf::LuaRef argmax(lua_State* L);

    /// Legacy 分类 Top-K
    LuaIntf::LuaRef topk(lua_State* L, int k);


private:
    // ==================== 成员变量 ====================

    std::shared_ptr<DeviceBuffer> buffer_;    ///< 底层缓冲区（抽象多设备）
    std::vector<int64_t> shape_;              ///< 逻辑形状
    std::vector<int64_t> strides_;            ///< 内存步长（支持非连续）
    int64_t offset_;                          ///< 数据起始偏移
    bool contiguous_;                         ///< 连续性标志
    DeviceType device_cache_;                 ///< 设备类型缓存（OPT-4）


    // ==================== 内部辅助方法 ====================
    // 实现文件: tensor_core.cpp

    /// 计算多维索引的线性偏移
    int64_t compute_offset(const std::vector<int64_t>& indices) const;

    /// 计算总元素数
    int64_t compute_size() const;

    /// 计算步长数组
    std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape) const;

    /// 标准化轴索引（处理负数）
    void normalize_axis(int& axis) const;

    // 实现文件: tensor_device.cpp
    /// 创建连续副本（优化批量 memcpy - OPT-6）
    Tensor contiguous_copy() const;

    /// 检查是否在 CPU 上（内联热路径 - OPT-2）
    inline void check_cpu() const {
        if (__builtin_expect(device_cache_ != DeviceType::CPU, 0)) {
            throw std::runtime_error("Operation requires CPU tensor");
        }
    }
};

} // namespace tensor
