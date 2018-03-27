/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer/activation.h"

namespace keras {
namespace layers {

class LSTM : public Layer {
public:
    LSTM() : return_sequences_(false) {}
    ~LSTM() override {}
    bool load_layer(std::ifstream* file) override;
    bool apply(Tensor* in, Tensor* out) override;

private:
    bool step(Tensor* x, Tensor* out, Tensor* ht_1, Tensor* ct_1);

    Tensor Wi_;
    Tensor Ui_;
    Tensor bi_;
    Tensor Wf_;
    Tensor Uf_;
    Tensor bf_;
    Tensor Wc_;
    Tensor Uc_;
    Tensor bc_;
    Tensor Wo_;
    Tensor Uo_;
    Tensor bo_;

    Activation inner_activation_;
    Activation activation_;
    bool return_sequences_;
};

} // namespace layers
} // namespace keras
