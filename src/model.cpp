/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
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

namespace keras {

namespace layers {

template <typename T>
std::unique_ptr<LayerBase> make(Stream& file) {
    return std::make_unique<T>(file);
}
template <typename... T>
class Factory {
public:
    std::unique_ptr<LayerBase> operator()(unsigned index, Stream& file) {
        static constexpr std::unique_ptr<LayerBase> (*factories[])(Stream&) = {
            &make<T>...
        };
        if (sizeof...(T) <= index) {
            throw std::range_error("type index out of range");
        }
        return (factories[index])(file);
    }
};

using factory = Factory<
    Dense,               //0
    Conv1D,              //1
    Conv2D,              //2
    LocallyConnected1D,  //3
    LocallyConnected2D,  //4
    Flatten,             //5
    ELU,                 //6
    Activation,          //7
    MaxPooling2D,        //8
    LSTM,                //9
    Embedding,           //10
    BatchNormalization>; //11

} // namespace layers

template <size_t... I>
std::unique_ptr<LayerBase>
_make_layer(Stream& file) {
    auto id = static_cast<unsigned>(file);
    return layers::factory()(id, file);
}

Model::Model(Stream& file) {
    auto count = static_cast<unsigned>(file);
    layers_.reserve(count);
    for (size_t i = 0; i != count; ++i)
        layers_.push_back(_make_layer(file));
}

Tensor Model::forward(const Tensor& in) const noexcept {
    Tensor out = in;
    for (auto&& layer : layers_)
        out = layer->forward(out);
    return out;
}

} // namespace keras
