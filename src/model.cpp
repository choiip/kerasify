/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/model.h"
#include "keras/layer/conv2d.h"
#include "keras/layer/dense.h"
#include "keras/layer/elu.h"
#include "keras/layer/embedding.h"
#include "keras/layer/flatten.h"
#include "keras/layer/lstm.h"
#include "keras/layer/maxpooling2d.h"
#include <cmath>
#include <limits>
#include <utility>

bool keras_model::load_model(const std::string& filename)
{
    std::ifstream file(filename.c_str(), std::ios::binary);
    check(file.is_open());

    unsigned num_layers = 0;
    check(read_uint(&file, num_layers));

    for (size_t i = 0; i < num_layers; ++i) {
        unsigned layer_type = 0;
        check(read_uint(&file, layer_type));

        keras_layer* layer = nullptr;

        switch (layer_type) {
        case kDense:
            layer = new keras_layer_dense();
            break;
        case kConvolution2d:
            layer = new keras_layer_conv2d();
            break;
        case kFlatten:
            layer = new keras_layer_flatten();
            break;
        case kElu:
            layer = new keras_layer_elu();
            break;
        case kActivation:
            layer = new keras_layer_activation();
            break;
        case kMaxPooling2D:
            layer = new keras_layer_maxpooling2d();
            break;
        case kLSTM:
            layer = new keras_layer_lstm();
            break;
        case kEmbedding:
            layer = new keras_layer_embedding();
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
        layers_.push_back(layer);
    }
    return true;
}

bool keras_model::apply(tensor* in, tensor* out)
{
    tensor temp_in, temp_out;
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (i == 0)
            temp_in = *in;
        check(!layers_[i]->apply(&temp_in, &temp_out));
        temp_in = temp_out;
    }
    *out = temp_out;
    return true;
}
