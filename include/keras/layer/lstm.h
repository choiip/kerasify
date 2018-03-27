/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer/activation.h"

class keras_layer_lstm : public keras_layer {
public:
    keras_layer_lstm() : return_sequences_(false) {}
    virtual ~keras_layer_lstm() {}
    virtual bool load_layer(std::ifstream* file);
    virtual bool apply(tensor* in, tensor* out);

private:
    bool step(tensor* x, tensor* out, tensor* ht_1, tensor* ct_1);

    tensor Wi_;
    tensor Ui_;
    tensor bi_;
    tensor Wf_;
    tensor Uf_;
    tensor bf_;
    tensor Wc_;
    tensor Uc_;
    tensor bc_;
    tensor Wo_;
    tensor Uo_;
    tensor bo_;

    keras_layer_activation inner_activation_;
    keras_layer_activation activation_;
    bool return_sequences_;
};
