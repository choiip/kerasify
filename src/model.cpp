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
    kassert(file);

    auto make_layer = [](unsigned layer_type) -> std::unique_ptr<Layer> {
        switch (layer_type) {
        case Dense:
            printf(" add Dense layer\n");
            return std::make_unique<layers::Dense>();
        case Conv1D:
            printf(" add Conv1D layer\n");
            return std::make_unique<layers::Conv1D>();
        case Conv2D:
            printf(" add Conv2D layer\n");
            return std::make_unique<layers::Conv2D>();
        case LocallyConnected1D:
            printf(" add LocallyConnected1D layer\n");
            return std::make_unique<layers::LocallyConnected1D>();
        case LocallyConnected2D:
            printf(" add LocallyConnected2D layer\n");
            return std::make_unique<layers::LocallyConnected2D>();
        case Flatten:
            printf(" add Flatten layer\n");
            return std::make_unique<layers::Flatten>();
        case ELU:
            printf(" add ELU layer\n");
            return std::make_unique<layers::ELU>();
        case Activation:
            printf(" add Activation layer\n");
            return std::make_unique<layers::Activation>();
        case MaxPooling2D:
            printf(" add MaxPooling2D layer\n");
            return std::make_unique<layers::MaxPooling2D>();
        case LSTM:
            printf(" add LSTM layer\n");
            return std::make_unique<layers::LSTM>();
        case Embedding:
            printf(" add Embedding layer\n");
            return std::make_unique<layers::Embedding>();
        case BatchNormalization:
            printf(" add BatchNormalization layer\n");
            return std::make_unique<layers::BatchNormalization>();
        }
        printf(" unknown layer\n");
        kassert(false);
        return nullptr;
    };

    auto layers_count = file.get<unsigned>();
    layers_.reserve(layers_count);
    for (auto i = 0; i != layers_count; ++i) {
        auto layer = make_layer(file.get<unsigned>());
        layer->load(file);
        layers_.push_back(layer);
    }
}

Tensor Model::operator()(const Tensor& in) const noexcept {
    Tensor out = in;
    for (auto&& layer : layers_)
        out = (*layer)(out);
    return out;
}

} // namespace keras
