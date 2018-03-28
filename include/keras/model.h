/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {

class Model {
    // TODO: rewrite using smart pointers
    std::vector<Layer*> layers_;

public:
    enum layer_type {
        Dense = 1,
        Conv2D = 2,
        Flatten = 3,
        ELU = 4,
        Activation = 5,
        MaxPooling2D = 6,
        LSTM = 7,
        Embedding = 8
    };

    Model() {}

    virtual ~Model()
    {
        for (auto&& it : layers_)
            delete it;
    }

    virtual bool load_model(const std::string& filename);
    virtual bool apply(const Tensor& in, Tensor& out);
};

} // namespace keras
