#pragma once

namespace node {

// Error codes compatible with sscma-node
constexpr int MA_OK       = 0;   // Success
constexpr int MA_ENOENT   = 2;   // Not found
constexpr int MA_EIO      = 5;   // I/O error
constexpr int MA_EAGAIN   = 11;  // Try again (deps not ready)
constexpr int MA_ENOMEM   = 12;  // Out of memory
constexpr int MA_EBUSY    = 16;  // Resource busy
constexpr int MA_EEXIST   = 17;  // Already exists
constexpr int MA_EINVAL   = 22;  // Invalid argument

} // namespace node
