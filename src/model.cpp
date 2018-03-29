/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/model.h"
#include "keras/layers/conv1d.h"
#include "keras/layers/conv2d.h"
#include "keras/layers/locally1d.h"
#include "keras/layers/locally2d.h"
#include "keras/layers/dense.h"
#include "keras/layers/elu.h"
#include "keras/layers/embedding.h"
#include "keras/layers/flatten.h"
#include "keras/layers/lstm.h"
#include "keras/layers/maxpooling2d.h"
#include "keras/layers/normalization.h"
#include <limits>
#include <utility>

namespace keras {
bool Model::load_model(const std::string& filename)
{
    std::ifstream file(filename.c_str(), std::ios::binary);
    check(file.is_open());

    unsigned num_layers = 0;
    check(read_uint(&file, num_layers));

    for (size_t i = 0; i < num_layers; ++i) {
        unsigned layer_type = 0;
        check(read_uint(&file, layer_type));

        Layer* layer = nullptr;

        switch (layer_type) {
        case Dense:
            layer = new layers::Dense();
            break;
        case Conv1D:
            layer = new layers::Conv1D();
            break;
        case Conv2D:
            layer = new layers::Conv2D();
            break;
        case LocallyConnected1D:
            layer = new layers::LocallyConnected1D();
            break;
        case LocallyConnected2D:
            layer = new layers::LocallyConnected2D();
            break;
        case Flatten:
            layer = new layers::Flatten();
            break;
        case ELU:
            layer = new layers::ELU();
            break;
        case Activation:
            layer = new layers::Activation();
            break;
        case MaxPooling2D:
            layer = new layers::MaxPooling2D();
            break;
        case LSTM:
            layer = new layers::LSTM();
            break;
        case Embedding:
            layer = new layers::Embedding();
            break;
        case BatchNormalization:
            layer = new layers::BatchNormalization();
            break;
        default:
            break;
        }
        check(layer);

        bool result = layer->load_layer(&file);
        if (!result) {
            delete layer;
            return false;
        }
        layers_.emplace_back(layer);
    }
    return true;
}

bool Model::apply(const Tensor& in, Tensor& out) const
{
    Tensor temp_in, temp_out;
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (i == 0)
            temp_in = in;
        check(layers_[i]->apply(temp_in, temp_out));
        temp_in = temp_out;
    }
    out = temp_out;
    return true;
}

} // namespace keras
