/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/lstm.h"

namespace keras {
namespace layers {

bool LSTM::load_layer(std::ifstream& file) noexcept
{
    unsigned wi_rows = 0;
    check(read_uint(file, wi_rows));
    check(wi_rows > 0);

    unsigned wi_cols = 0;
    check(read_uint(file, wi_cols));
    check(wi_cols > 0);

    unsigned ui_rows = 0;
    check(read_uint(file, ui_rows));
    check(ui_rows > 0);

    unsigned ui_cols = 0;
    check(read_uint(file, ui_cols));
    check(ui_cols > 0);

    unsigned bi_shape = 0;
    check(read_uint(file, bi_shape));
    check(bi_shape > 0);

    unsigned wf_rows = 0;
    check(read_uint(file, wf_rows));
    check(wf_rows > 0);

    unsigned wf_cols = 0;
    check(read_uint(file, wf_cols));
    check(wf_cols > 0);

    unsigned uf_rows = 0;
    check(read_uint(file, uf_rows));
    check(uf_rows > 0);

    unsigned uf_cols = 0;
    check(read_uint(file, uf_cols));
    check(uf_cols > 0);

    unsigned bf_shape = 0;
    check(read_uint(file, bf_shape));
    check(bf_shape > 0);

    unsigned wc_rows = 0;
    check(read_uint(file, wc_rows));
    check(wc_rows > 0);

    unsigned wc_cols = 0;
    check(read_uint(file, wc_cols));
    check(wc_cols > 0);

    unsigned uc_rows = 0;
    check(read_uint(file, uc_rows));
    check(uc_rows > 0);

    unsigned uc_cols = 0;
    check(read_uint(file, uc_cols));
    check(uc_cols > 0);

    unsigned bc_shape = 0;
    check(read_uint(file, bc_shape));
    check(bc_shape > 0);

    unsigned wo_rows = 0;
    check(read_uint(file, wo_rows));
    check(wo_rows > 0);

    unsigned wo_cols = 0;
    check(read_uint(file, wo_cols));
    check(wo_cols > 0);

    unsigned uo_rows = 0;
    check(read_uint(file, uo_rows));
    check(uo_rows > 0);

    unsigned uo_cols = 0;
    check(read_uint(file, uo_cols));
    check(uo_cols > 0);

    unsigned bo_shape = 0;
    check(read_uint(file, bo_shape));
    check(bo_shape > 0);

    // Load Input Weights and Biases
    Wi_.resize(wi_rows, wi_cols);
    check(read_floats(file, Wi_.data_.data(), wi_rows * wi_cols));

    Ui_.resize(ui_rows, ui_cols);
    check(read_floats(file, Ui_.data_.data(), ui_rows * ui_cols));

    bi_.resize(1, bi_shape);
    check(read_floats(file, bi_.data_.data(), bi_shape));

    // Load Forget Weights and Biases
    Wf_.resize(wf_rows, wf_cols);
    check(read_floats(file, Wf_.data_.data(), wf_rows * wf_cols));

    Uf_.resize(uf_rows, uf_cols);
    check(read_floats(file, Uf_.data_.data(), uf_rows * uf_cols));

    bf_.resize(1, bf_shape);
    check(read_floats(file, bf_.data_.data(), bf_shape));

    // Load State Weights and Biases
    Wc_.resize(wc_rows, wc_cols);
    check(read_floats(file, Wc_.data_.data(), wc_rows * wc_cols));

    Uc_.resize(uc_rows, uc_cols);
    check(read_floats(file, Uc_.data_.data(), uc_rows * uc_cols));

    bc_.resize(1, bc_shape);
    check(read_floats(file, bc_.data_.data(), bc_shape));

    // Load Output Weights and Biases
    Wo_.resize(wo_rows, wo_cols);
    check(read_floats(file, Wo_.data_.data(), wo_rows * wo_cols));

    Uo_.resize(uo_rows, uo_cols);
    check(read_floats(file, Uo_.data_.data(), uo_rows * uo_cols));

    bo_.resize(1, bo_shape);
    check(read_floats(file, bo_.data_.data(), bo_shape));

    check(inner_activation_.load_layer(file));
    check(activation_.load_layer(file));

    unsigned return_sequences = 0;
    check(read_uint(file, return_sequences));
    return_sequences_ = static_cast<bool>(return_sequences);
    return true;
}

bool LSTM::apply(const Tensor& in, Tensor& out) const noexcept
{
    // Assume 'bo_' always keeps the output shape and we will always
    // receive one single sample.
    size_t out_dim = bo_.dims_[1];
    size_t steps = in.dims_[0];

    Tensor ht_1{1, out_dim};
    Tensor ct_1{1, out_dim};

    ht_1.fill(0.f);
    ct_1.fill(0.f);

    if (!return_sequences_) {
        for (size_t s = 0; s < steps; ++s)
            check(step(in.select(s), out, ht_1, ct_1));
        return true;
    }

    out.dims_ = {steps, out_dim};
    out.data_.reserve(steps * out_dim);

    Tensor last;
    for (size_t s = 0; s < steps; ++s) {
        check(step(in.select(s), last, ht_1, ct_1));
        out.data_.insert(out.end(), last.begin(), last.end());
    }
    return true;
}

bool LSTM::step(const Tensor& x, Tensor& out, Tensor& ht_1, Tensor& ct_1) const
    noexcept
{
    Tensor xi = x.dot(Wi_) + bi_;
    Tensor xf = x.dot(Wf_) + bf_;
    Tensor xc = x.dot(Wc_) + bc_;
    Tensor xo = x.dot(Wo_) + bo_;

    Tensor i_ = xi + ht_1.dot(Ui_);
    Tensor f_ = xf + ht_1.dot(Uf_);
    Tensor c_ = xc + ht_1.dot(Uc_);
    Tensor o_ = xo + ht_1.dot(Uo_);

    Tensor i, f, cc, o;

    check(inner_activation_.apply(i_, i));
    check(inner_activation_.apply(f_, f));
    check(activation_.apply(c_, cc));
    check(inner_activation_.apply(o_, o));

    ct_1 = f.multiply(ct_1) + i.multiply(cc);

    check(activation_.apply(ct_1, cc));

    out = ht_1 = o.multiply(cc);
    return true;
}

} // namespace layers
} // namespace keras
