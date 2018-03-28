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

    check(in->dims_[2] == weights_.dims_[3]);

    size_t st_nj = (weights_.dims_[1] - 1) / 2;
    size_t st_pj = (weights_.dims_[1]) / 2;
    size_t st_nk = (weights_.dims_[2] - 1) / 2;
    size_t st_pk = (weights_.dims_[2]) / 2;

    Tensor tmp{in->dims_[0] - st_nj - st_pj, in->dims_[1] - st_nk - st_pk,
               weights_.dims_[3]};

    // Iterate over each kernel.
    for (size_t k = 0; k < weights_.dims_[0]; ++k) {
        // 2D convolution in x and y (k and l in Tensor dimensions).
        for (size_t y = 0; y < tmp.dims_[0]; ++y)
            for (size_t x = 0; x < tmp.dims_[1]; ++x)
                // Iterate over kernel.
                for (size_t ky = 0; ky < weights_.dims_[1]; ++ky)
                    for (size_t kx = 0; kx < weights_.dims_[2]; ++kx)
                        for (size_t kd = 0; kd < weights_.dims_[3]; ++kd) {
                            const float& weight = weights_(k, ky, kx, kd);
                            const float& value = (*in)(y + ky, x + kx, kd);
                            tmp(k, y, x) += weight * value;
                        }
        for (size_t y = 0; y < tmp.dims_[0]; ++y)
            for (size_t x = 0; x < tmp.dims_[1]; ++x)
                tmp(k, y, x) += biases_(k);
    }
    check(activation_.apply(&tmp, out));
    return true;
}

} // namespace layers
} // namespace keras
