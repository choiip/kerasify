/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layer/conv2d.h"

namespace keras {
namespace layers {

bool Conv2D::load_layer(std::ifstream* file)
{
    check(file);

    unsigned weights_i = 0;
    check(read_uint(file, weights_i));
    check(weights_i > 0);

    unsigned weights_j = 0;
    check(read_uint(file, weights_j));
    check(weights_j > 0);

    unsigned weights_k = 0;
    check(read_uint(file, weights_k));
    check(weights_k > 0);

    unsigned weights_l = 0;
    check(read_uint(file, weights_l));
    check(weights_l > 0);

    unsigned biases_shape = 0;
    check(read_uint(file, biases_shape));
    check(biases_shape > 0);

    weights_.resize(weights_i, weights_j, weights_k, weights_l);
    check(read_floats(
        file, weights_.data_.data(),
        weights_i * weights_j * weights_k * weights_l));

    biases_.resize(biases_shape);
    check(read_floats(file, biases_.data_.data(), biases_shape));

    check(activation_.load_layer(file));
    return true;
}

bool Conv2D::apply(Tensor* in, Tensor* out)
{
    check(in);
    check(out);

    check(in->dims_[0] == weights_.dims_[1]);

    size_t st_nj = (weights_.dims_[2] - 1) / 2;
    size_t st_pj = (weights_.dims_[2]) / 2;
    size_t st_nk = (weights_.dims_[3] - 1) / 2;
    size_t st_pk = (weights_.dims_[3]) / 2;

    Tensor tmp{weights_.dims_[0], in->dims_[1] - st_nj - st_pj,
               in->dims_[2] - st_nk - st_pk};

    for (size_t i = 0; i < weights_.dims_[0]; ++i) {
        for (size_t j = 0; j < weights_.dims_[1]; ++j)
            for (size_t tj = st_nj; tj < in->dims_[1] - st_pj; ++tj)
                for (size_t tk = st_nk; tk < in->dims_[2] - st_pk; ++tk)
                    for (size_t k = 0; k < weights_.dims_[2]; ++k)
                        for (size_t l = 0; l < weights_.dims_[3]; ++l) {
                            const float& weight = weights_(i, j, k, l);
                            const float& value =
                                (*in)(j, tj - st_nj + k, tk - st_nk + l);
                            tmp(i, tj - st_nj, tk - st_nk) += weight * value;
                        }
        for (size_t j = 0; j < tmp.dims_[1]; ++j)
            for (size_t k = 0; k < tmp.dims_[2]; ++k)
                tmp(i, j, k) += biases_(i);
    }
    check(activation_.apply(&tmp, out));
    return true;
}

} // namespace layers
} // namespace keras
