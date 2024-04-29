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
        "bias or residual is not supported in CPU rms_norm yet"));
  }
  if (quant_scale > 0.0f) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "quantization is not supported in CPU rms_norm yet"));
  }

  // init output
  dev_ctx.template Alloc<T>(out);
  if (inv_var) {
    dev_ctx.template Alloc<T>(inv_var);
  }

  const auto& x_dims = x.dims();
  auto matrix_dim = common::flatten_to_2d(x_dims, begin_norm_axis);
  int rows = static_cast<int>(matrix_dim[0]);
  int cols = static_cast<int>(matrix_dim[1]);
  DDim matrix_shape({rows, cols});
  DDim var_shape({rows});

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
}
}  // namespace phi

PD_REGISTER_KERNEL(rms_norm,
                   CPU,
                   ALL_LAYOUT,
                   phi::RmsNormKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
