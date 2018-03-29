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
        Conv1D = 2,
        Conv2D = 3,
        Flatten = 4,
        ELU = 5,
        Activation = 6,
        MaxPooling2D = 7,
        LSTM = 8,
        Embedding = 9,
        BatchNormalization = 10,
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
