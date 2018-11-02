/*
 * Copyright (c) 2016 Robert W. Rose, 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layers/activation.h"

namespace keras {
namespace layers {

class LSTM final : public Layer {
public:
    void load(Stream& file) override;
    Tensor operator()(const Tensor& in) const noexcept override;

private:
    std::tuple<Tensor, Tensor>
    step(const Tensor& x, const Tensor& ht_1, const Tensor& ct_1)
         const noexcept;

    Tensor Wi_;
    Tensor Ui_;
    Tensor bi_;
    Tensor Wf_;
    Tensor Uf_;
    Tensor bf_;
    Tensor Wc_;
    Tensor Uc_;
    Tensor bc_;
    Tensor Wo_;
    Tensor Uo_;
    Tensor bo_;

    Activation inner_activation_;
    Activation activation_;
    bool return_sequences_{false};
};

} // namespace layers
} // namespace keras
