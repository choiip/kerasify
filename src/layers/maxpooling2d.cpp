/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/maxpooling2d.h"

namespace keras {
namespace layers {

bool MaxPooling2D::load_layer(std::ifstream& file) noexcept
{
    check(read_uint(file, pool_size_y_));
    check(read_uint(file, pool_size_x_));
    return true;
}

bool MaxPooling2D::apply(const Tensor& in, Tensor& out) const noexcept
{
    check(in.dims_.size() == 3);

    out.resize(
        in.dims_[0] / pool_size_y_, in.dims_[1] / pool_size_x_, in.dims_[2]);
    out.fill(-std::numeric_limits<float>::infinity());

    const auto& iw = in.dims_;
    const auto& ow = out.dims_;

    size_t is0 = iw[1] * iw[2];
    size_t is1 = iw[2];

    size_t ip_ = is0 * pool_size_y_;
    size_t ip0 = is1 * pool_size_x_;

    size_t os_ = ow[0] * ow[1] * ow[2];
    size_t os0 = ow[1] * ow[2];
    size_t os1 = ow[2];

    auto* o_ptr = &out.data_[0];
    auto* i_ptr = &in.data_[0];
    for (auto* o__ = o_ptr; o__ < o_ptr + os_; o__ += os0, i_ptr += ip_) {
        auto* i_ = i_ptr;
        for (auto* o_ = o__; o_ < o__ + os0; o_ += os1, i_ += ip0) {
            for (auto* i0 = i_; i0 < i_ + ip_; i0 += is0)
                for (auto* i1 = i0; i1 < i0 + ip0; i1 += is1) {
                    auto* o1 = o_;
                    for (auto* i2 = i1; i2 < i1 + is1; ++i2) {
                        if (*i2 > *o1)
                            *o1 = *i2;
                        ++o1;
                    }
                }
        }
    }
    return true;
}

} // namespace layers
} // namespace keras
