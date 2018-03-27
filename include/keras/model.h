/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {

class Model {
    std::vector<Layer*> layers_;

public:
    enum layer_type {
        kDense = 1,
        kConvolution2d = 2,
        kFlatten = 3,
        kElu = 4,
        kActivation = 5,
        kMaxPooling2D = 6,
        kLSTM = 7,
        kEmbedding = 8
    };

    Model() {}

    virtual ~Model()
    {
        for (size_t i = 0; i < layers_.size(); ++i)
            delete layers_[i];
    }

    virtual bool load_model(const std::string& filename);
    virtual bool apply(Tensor* in, Tensor* out);
};

} // namespace keras
