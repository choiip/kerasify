/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/io.h"
#include "keras/tensor.h"

#include <string>

namespace keras {

class LayerBase {
public:
    LayerBase() = default;
    LayerBase(Stream&) : LayerBase() {}

    virtual ~LayerBase();
    virtual Tensor forward(Tensor const& in) const noexcept = 0;
};

template <typename Derived>
class Layer : public LayerBase {
public:
    using LayerBase::LayerBase;

    static Derived load(std::string const& filename) {
        Stream file {filename};
        return {file};
    }

    Tensor operator()(Tensor const& in) const {
        return static_cast<Derived const*>(this)->forward(in);
    }
};

} // namespace keras
