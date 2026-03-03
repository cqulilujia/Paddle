// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/bitwise_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/logical_kernel.h"

namespace phi {

template <typename T, typename Context>
void BitwiseNotKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      DenseTensor* out) {
  using XPUDataType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(out);
  int r = xpu::logical_not(dev_ctx.x_context(),
                           reinterpret_cast<const XPUDataType*>(x.data<T>()),
                           reinterpret_cast<XPUDataType*>(out->data<T>()),
                           x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "logical_not");
}

template <typename T, typename Context>
void BitwiseAndKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      DenseTensor* out) {
  using XPUDataType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(out);
  std::vector<std::int64_t> xshape(x.dims().size());
  for (int i = 0; i < x.dims().size(); ++i) {
    xshape[i] = static_cast<std::int64_t>(x.dims()[i]);
  }

  std::vector<std::int64_t> yshape(y.dims().size());
  for (int i = 0; i < y.dims().size(); ++i) {
    yshape[i] = static_cast<std::int64_t>(y.dims()[i]);
  }
  int r =
      xpu::bitwise_and_tensor(dev_ctx.x_context(),
                              reinterpret_cast<const XPUDataType*>(x.data<T>()),
                              reinterpret_cast<const XPUDataType*>(y.data<T>()),
                              reinterpret_cast<XPUDataType*>(out->data<T>()),
                              xshape,
                              yshape);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "bitwise_and_tensor");
}

template <typename T, typename Context>
void BitwiseOrKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  // Same reason as bitwise_and
  LogicalOrKernel<T, Context>(dev_ctx, x, y, out);
}
}  // namespace phi

PD_REGISTER_KERNEL(bitwise_not, XPU, ALL_LAYOUT, phi::BitwiseNotKernel, bool) {}
PD_REGISTER_KERNEL(bitwise_and,
                   XPU,
                   ALL_LAYOUT,
                   phi::BitwiseAndKernel,
                   bool,
                   int64_t,
                   int,
                   int16_t,
                   int8_t,
                   uint8_t) {}
PD_REGISTER_KERNEL(bitwise_or, XPU, ALL_LAYOUT, phi::BitwiseOrKernel, bool) {}
