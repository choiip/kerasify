/*
 * Copyright (c) 2016 Robert W. Rose, 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/lstm.h"
#include <tuple>

namespace keras {
namespace layers {

void LSTM::load(Stream& file) noexcept {
    // Load Input Weights and Biases
    Wi_.load(file, 2);
    Ui_.load(file, 2);
    bi_.load(file, 2);

    // Load Forget Weights and Biases
    Wf_.load(file, 2);
    Uf_.load(file, 2);
    bf_.load(file, 2);

    // Load State Weights and Biases
    Wc_.load(file, 2);
    Uc_.load(file, 2);
    bc_.load(file, 2);

    // Load Output Weights and Biases
    Wo_.load(file, 2);
    Uo_.load(file, 2);
    bo_.load(file, 2);

    inner_activation_.load(file);
    activation_.load(file);

    return_sequences_ = static_cast<bool>(file.to_uint());
}

Tensor LSTM::operator()(const Tensor& in) const noexcept {
    // Assume 'bo_' always keeps the output shape and we will always
    // receive one single sample.
    size_t out_dim = bo_.dims_[1];
    size_t steps = in.dims_[0];

    Tensor c_tm1{1, out_dim};
    c_tm1.fill(0.f);

    if (!return_sequences_) {
        Tensor out{1, out_dim};
        out.fill(0.f);
        for (size_t s = 0; s < steps; ++s)
            std::tie(out, c_tm1) = step(in.select(s), out, c_tm1);
        return out;
    }

    auto out = Tensor::empty(steps, out_dim);

    Tensor last{1, out_dim};
    last.fill(0.f);
    for (size_t s = 0; s < steps; ++s) {
        std::tie(last, c_tm1) = step(in.select(s), last, c_tm1);
        out.data_.insert(out.end(), last.begin(), last.end());
    }
    return out;
}

std::tuple<Tensor, Tensor>
LSTM::step(const Tensor& x, const Tensor& h_tm1, const Tensor& c_tm1)
           const noexcept {
    auto i_ = x.dot(Wi_) + h_tm1.dot(Ui_) + bi_;
    auto f_ = x.dot(Wf_) + h_tm1.dot(Uf_) + bf_;
    auto c_ = x.dot(Wc_) + h_tm1.dot(Uc_) + bc_;
    auto o_ = x.dot(Wo_) + h_tm1.dot(Uo_) + bo_;

    auto cc = activation_((inner_activation_(f_) * c_tm1) +
                          (inner_activation_(i_) * activation_(c_)));
    auto out = inner_activation_(o_) * activation_(cc);
    return std::make_tuple(out, cc);
}

} // namespace layers
} // namespace keras
