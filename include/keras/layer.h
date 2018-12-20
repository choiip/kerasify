/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/tensor.h"

namespace keras {

class BaseLayer {
public:
    BaseLayer() = default;
    BaseLayer(Stream&) : BaseLayer() {}

    BaseLayer(BaseLayer&&) = default;
    BaseLayer& operator=(BaseLayer&&) = default;

    virtual ~BaseLayer();
    virtual Tensor operator()(const Tensor& in) const noexcept = 0;
};

template <typename Derived>
class Layer : public BaseLayer {
public:
    using BaseLayer::BaseLayer;

    static Derived load(const std::string& filename) {
        Stream file(filename);
        return Derived(file);
    }
};

} // namespace keras
