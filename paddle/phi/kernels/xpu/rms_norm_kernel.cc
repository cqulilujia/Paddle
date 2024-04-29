// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/rms_norm_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RmsNormKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const paddle::optional<DenseTensor>& bias,
                   const paddle::optional<DenseTensor>& residual,
                   const DenseTensor& norm_weight,
                   const paddle::optional<DenseTensor>& norm_bias,
                   const float epsilon,
                   const int begin_norm_axis,
                   const float quant_scale,
                   const int quant_round_type,
                   const float quant_max_bound,
                   const float quant_min_bound,
                   DenseTensor* out,
                   DenseTensor* residual_out,
                   DenseTensor* inv_var) {
  if (bias || residual) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "bias or residual is not supported in XPU rms_norm yet"));
  }
  if (quant_scale > 0.0f) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "quantization is not supported in XPU rms_norm yet"));
  }
  using XPUType = typename XPUTypeTrait<T>::Type;

  const T* x_data = x.data<T>();
  const T* norm_weight_data = norm_weight.data<T>();
  const T* norm_bias_data = norm_bias ? norm_bias.get().data<T>() : nullptr;
  dev_ctx.template Alloc<T>(out);
  T* out_data = out->data<T>();
  float* inv_var_data = nullptr;
  if (inv_var != nullptr) {
    dev_ctx.template Alloc<float>(inv_var);
    inv_var_data = inv_var->data<float>();
  }

  int32_t rows = 1;
  int32_t cols = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    rows *= x.dims()[i];
  }

  for (int i = begin_norm_axis; i < x.dims().size(); i++) {
    cols *= x.dims()[i];
  }

  PADDLE_ENFORCE_EQ(
      cols,
      norm_weight.dims()[0],
      phi::errors::InvalidArgument(
          "The product from begin_norm_axis to the last axis of input tensor "
          "x, "
          "i.e., cols(%d)"
          "must be equal to the norm_weight tensor's dimension(%d).",
          cols,
          norm_weight.dims()[0]))

  int r = baidu::xpu::api::rms_layer_norm<XPUType, XPUType>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(x_data),
      reinterpret_cast<XPUType*>(out_data),
      rows,
      cols,
      epsilon,
      reinterpret_cast<const XPUType*>(norm_weight_data),
      reinterpret_cast<const XPUType*>(norm_bias_data),
      inv_var_data,
      true);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "rms_layer_norm");
}
}  // namespace phi

PD_REGISTER_KERNEL(rms_norm,
                   XPU,
                   ALL_LAYOUT,
                   phi::RmsNormKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
