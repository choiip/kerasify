/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layer/maxpooling2d.h"

namespace keras {
namespace layers {

bool MaxPooling2D::load_layer(std::ifstream* file)
{
    check(file);
    check(read_uint(file, pool_size_y_));
    check(read_uint(file, pool_size_x_));
    return true;
}

bool MaxPooling2D::apply(Tensor* in, Tensor* out)
{
    check(in);
    check(out);
    check(in->dims_.size() == 3);

    Tensor tmp(
        in->dims_[0] / pool_size_y_, in->dims_[1] / pool_size_x_, in->dims_[2]);

    for (size_t i = 0; i < tmp.dims_[2]; ++i)
        for (size_t y2 = 0; y2 < tmp.dims_[0]; ++y2) {
            const size_t y1 = y2 * pool_size_y_;
            for (size_t x2 = 0; x2 < tmp.dims_[1]; ++x2) {
                const size_t x1 = x2 * pool_size_x_;
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t py = 0; py < pool_size_y_; ++py)
                    for (size_t px = 0; px < pool_size_x_; ++px) {
                        const float& pool_val = (*in)(y1 + py, x1 + px, i);
                        if (pool_val > max_val)
                            max_val = pool_val;
                    }
                tmp(y2, x2, i) = max_val;
            }
        }
    *out = tmp;
    return true;
}

} // namespace layers
} // namespace keras
