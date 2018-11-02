/*
 * Copyright (c) 2016 Robert W. Rose, 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class ELU final : public Layer {
public:
    void load(Stream& file) override;
    Tensor operator()(const Tensor& in) const noexcept override;

private:
    float alpha_{1.f};
};

} // namespace layers
} // namespace keras
