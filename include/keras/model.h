/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/io.h"
#include "keras/layer.h"

#include <memory>
#include <vector>

namespace keras {

class Model final : public Layer<Model> {
    std::vector<std::unique_ptr<LayerBase>> layers_;

public:
    Model(Stream& file);
    Tensor operator()(const Tensor& in) const noexcept override;
};

} // namespace keras
