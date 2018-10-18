/*
 * Copyright (c) 2016 Robert W. Rose, 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/model.h"
#include "keras/layers/conv1d.h"
#include "keras/layers/conv2d.h"
#include "keras/layers/dense.h"
#include "keras/layers/elu.h"
#include "keras/layers/embedding.h"
#include "keras/layers/flatten.h"
#include "keras/layers/locally1d.h"
#include "keras/layers/locally2d.h"
#include "keras/layers/lstm.h"
#include "keras/layers/maxpooling2d.h"
#include "keras/layers/normalization.h"
#include <limits>
#include <utility>

namespace keras {

void Model::load(const std::string& filename) noexcept {
    Stream file(filename, std::ios::binary);
    kassert(file.is_open());

    layers_.reserve(file.to_uint());

    auto make_layer = [](unsigned layer_type) -> std::unique_ptr<Layer> {
        switch (layer_type) {
        case Dense:
            return std::make_unique<layers::Dense>();
        case Conv1D:
            return std::make_unique<layers::Conv1D>();
        case Conv2D:
            return std::make_unique<layers::Conv2D>();
        case LocallyConnected1D:
            return std::make_unique<layers::LocallyConnected1D>();
        case LocallyConnected2D:
            return std::make_unique<layers::LocallyConnected2D>();
        case Flatten:
            return std::make_unique<layers::Flatten>();
        case ELU:
            return std::make_unique<layers::ELU>();
        case Activation:
            return std::make_unique<layers::Activation>();
        case MaxPooling2D:
            return std::make_unique<layers::MaxPooling2D>();
        case LSTM:
            return std::make_unique<layers::LSTM>();
        case Embedding:
            return std::make_unique<layers::Embedding>();
        case BatchNormalization:
            return std::make_unique<layers::BatchNormalization>();
        }
        kassert(false);
        return nullptr;
    };

    std::generate(layers_.begin(), layers_.end(), [&file, &make_layer]{
        auto layer = make_layer(file.to_uint());
        layer->load(file);
        return layer;
    });
}

Tensor Model::operator()(const Tensor& in) const noexcept {
    Tensor out = in;
    for (auto&& layer : layers_)
        out = (*layer)(out);
    return out;
}

} // namespace keras
