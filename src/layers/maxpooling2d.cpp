/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/maxpooling2d.h"

namespace keras {
namespace layers {

bool MaxPooling2D::load_layer(std::ifstream* file)
{
    check(file);
    check(read_uint(file, pool_size_y_));
    check(read_uint(file, pool_size_x_));
    return true;
}

// TODO: optimize for speed
bool MaxPooling2D::apply(const Tensor& in, Tensor& out) const
{
    check(in.dims_.size() == 3);

    out.resize(
        in.dims_[0] / pool_size_y_, in.dims_[1] / pool_size_x_, in.dims_[2]);
    const auto& out_size = out.dims_;

    for (size_t y2 = 0; y2 < out_size[0]; ++y2) {
        const size_t y1 = y2 * pool_size_y_;
        for (size_t x2 = 0; x2 < out_size[1]; ++x2) {
            const size_t x1 = x2 * pool_size_x_;
            for (size_t ch = 0; ch < out_size[2]; ++ch) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t py = 0; py < pool_size_y_; ++py)
                    for (size_t px = 0; px < pool_size_x_; ++px) {
                        const float& pool_val = in(y1 + py, x1 + px, ch);
                        if (pool_val > max_val)
                            max_val = pool_val;
                    }
                out(y2, x2, ch) = max_val;
            }
        }
    }
    return true;
}

} // namespace layers
} // namespace keras
