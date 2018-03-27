/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layer/dense.h"

bool keras_layer_dense::load_layer(std::ifstream* file)
{
    check(file);

    unsigned weights_rows = 0;
    check(read_uint(file, weights_rows));
    check(weights_rows > 0);

    unsigned weights_cols = 0;
    check(read_uint(file, weights_cols));
    check(weights_cols > 0);

    unsigned biases_shape = 0;
    check(read_uint(file, biases_shape));
    check(biases_shape > 0);

    weights_.resize(weights_rows, weights_cols);
    check(
        read_floats(file, weights_.data_.data(), weights_rows * weights_cols));

    biases_.resize(biases_shape);
    check(read_floats(file, biases_.data_.data(), biases_shape));

    check(activation_.load_layer(file));
    return true;
}

bool keras_layer_dense::apply(tensor* in, tensor* out)
{
    check(in);
    check(out);

    check(in->dims_.size() <= 2);

    if (in->dims_.size() == 2)
        check(in->dims_[1] == weights_.dims_[0]);

    tensor tmp{weights_.dims_[1]};

    for (size_t i = 0; i < weights_.dims_[0]; ++i)
        for (size_t j = 0; j < weights_.dims_[1]; ++j)
            tmp(j) += (*in)(i)*weights_(i, j);

    for (size_t i = 0; i < biases_.dims_[0]; ++i)
        tmp(i) += biases_(i);

    check(activation_.apply(&tmp, out));
    return true;
}
