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

void Model::load(const std::string& filename) {
    Stream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open " + filename);

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
        return nullptr;
    };

    auto layers_count = file.get<unsigned>();
    layers_.reserve(layers_count);
    for (unsigned i = 0; i != layers_count; ++i) {
        auto layer = make_layer(file.get<unsigned>());
        layer->load(file);
        layers_.push_back(std::move(layer));
    }
}

Tensor Model::operator()(const Tensor& in) const noexcept {
    Tensor out = in;
    for (auto&& layer : layers_)
        out = (*layer)(out);
    return out;
}

} // namespace keras
