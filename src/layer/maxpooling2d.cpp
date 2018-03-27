/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layer/maxpooling2d.h"

bool keras_layer_maxpooling2d::load_layer(std::ifstream* file)
{
    check(file);
    check(read_uint(file, pool_size_j_));
    check(read_uint(file, pool_size_k_));
    return true;
}

bool keras_layer_maxpooling2d::apply(tensor* in, tensor* out)
{
    check(in);
    check(out);
    check(in->dims_.size() != 3);

    tensor tmp(
        in->dims_[0], in->dims_[1] / pool_size_j_, in->dims_[2] / pool_size_k_);

    for (size_t i = 0; i < tmp.dims_[0]; ++i)
        for (size_t j = 0; j < tmp.dims_[1]; ++j) {
            const size_t tj = j * pool_size_j_;
            for (size_t k = 0; k < tmp.dims_[2]; ++k) {
                const size_t tk = k * pool_size_k_;
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t pj = 0; pj < pool_size_j_; ++pj)
                    for (size_t pk = 0; pk < pool_size_k_; ++pk) {
                        const float& pool_val = (*in)(i, tj + pj, tk + pk);
                        if (pool_val > max_val)
                            max_val = pool_val;
                    }
                tmp(i, j, k) = max_val;
            }
        }
    *out = tmp;
    return true;
}
