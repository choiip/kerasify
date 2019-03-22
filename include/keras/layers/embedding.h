/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class Embedding final : public Layer<Embedding> {
    Tensor weights_;

public:
    Embedding(Stream& file);
    Tensor forward(const Tensor& in) const noexcept override;
};

} // namespace layers
} // namespace keras
