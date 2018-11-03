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

class ELU final : public Layer<ELU> {
    float alpha_{1.f};

public:
    ELU(Stream& file);
    Tensor operator()(const Tensor& in) const noexcept override;
};

} // namespace layers
} // namespace keras
