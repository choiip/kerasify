/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer_weights.h"

namespace keras {
namespace layers {

class Conv1D final : public Layer<Conv1D>, public LayerWeights {
public:
    using LayerWeights::LayerWeights;
    Tensor forward(const Tensor& in) const noexcept override;
};

} // namespace layers
} // namespace keras
