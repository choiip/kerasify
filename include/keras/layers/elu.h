/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class ELU : public Layer {
public:
    ELU() : alpha_(1.0f) {}
    ~ELU() override {}
    bool load_layer(std::ifstream& file) override;
    bool apply(const Tensor& in, Tensor& out) const override;

private:
    float alpha_;
};

} // namespace layers
} // namespace keras
