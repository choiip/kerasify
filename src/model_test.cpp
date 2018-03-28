#include "test/benchmark.h"
#include "test/conv_2x2.h"
#include "test/conv_3x3.h"
#include "test/conv_3x3x3.h"
#include "test/conv_hard_sigmoid_2x2.h"
#include "test/conv_sigmoid_2x2.h"
#include "test/conv_softplus_2x2.h"
#include "test/dense_10x1.h"
#include "test/dense_10x10.h"
#include "test/dense_10x10x10.h"
#include "test/dense_1x1.h"
#include "test/dense_2x2.h"
#include "test/dense_relu_10.h"
#include "test/dense_tanh_10.h"
#include "test/elu_10.h"
#include "test/embedding_64.h"
#include "test/lstm_simple_7x20.h"
#include "test/lstm_simple_stacked_16x9.h"
#include "test/lstm_stacked_64x83.h"
#include "test/maxpool2d_1x1.h"
#include "test/maxpool2d_2x2.h"
#include "test/maxpool2d_3x2x2.h"
#include "test/maxpool2d_3x3x3.h"
#include "test/relu_10.h"

#include <iostream>
#include <stdio.h>

using namespace keras;

namespace test {
bool basics()
{
    {
        const int i = 3;
        const int j = 5;
        const int k = 10;
        Tensor t{i, j, k};

        float c = 1.f;
        for (size_t ii = 0; ii < i; ++ii)
            for (size_t jj = 0; jj < j; ++jj)
                for (size_t kk = 0; kk < k; ++kk) {
                    t(ii, jj, kk) = c;
                    c += 1.f;
                }
        c = 1.f;
        size_t cc = 0;
        for (size_t ii = 0; ii < i; ++ii)
            for (size_t jj = 0; jj < j; ++jj)
                for (size_t kk = 0; kk < k; ++kk) {
                    check_eq(t(ii, jj, kk), c, 1e-9);
                    check_eq(t.data_[cc], c, 1e-9);
                    c += 1.f;
                    ++cc;
                }
    }
    {
        const size_t i = 2;
        const size_t j = 3;
        const size_t k = 4;
        const size_t l = 5;
        Tensor t{i, j, k, l};

        float c = 1.f;
        for (size_t ii = 0; ii < i; ++ii)
            for (size_t jj = 0; jj < j; ++jj)
                for (size_t kk = 0; kk < k; ++kk)
                    for (size_t ll = 0; ll < l; ++ll) {
                        t(ii, jj, kk, ll) = c;
                        c += 1.f;
                    }
        c = 1.f;
        size_t cc = 0;
        for (size_t ii = 0; ii < i; ++ii)
            for (size_t jj = 0; jj < j; ++jj)
                for (size_t kk = 0; kk < k; ++kk)
                    for (size_t ll = 0; ll < l; ++ll) {
                        check_eq(t(ii, jj, kk, ll), c, 1e-9);
                        check_eq(t.data_[cc], c, 1e-9);
                        c += 1.f;
                        ++cc;
                    }
    }
    {
        Tensor a{2, 2};
        Tensor b{2, 2};

        a.data_ = {1.0, 2.0, 3.0, 5.0};
        b.data_ = {2.0, 5.0, 4.0, 1.0};

        Tensor result = a + b;
        check(result.data_ == std::vector<float>({3.0, 7.0, 7.0, 6.0}));
    }
    {
        Tensor a{2, 2};
        Tensor b{2, 2};

        a.data_ = {1.0, 2.0, 3.0, 5.0};
        b.data_ = {2.0, 5.0, 4.0, 1.0};

        Tensor result = a.multiply(b);
        check(result.data_ == std::vector<float>({2.0, 10.0, 12.0, 5.0}));
    }
    {
        Tensor a{1, 2};
        Tensor b{2, 1};

        a.data_ = {1.0, 2.0};
        b.data_ = {2.0, 5.0};

        Tensor result = a.dot(b);
        check(result.data_ == std::vector<float>({12.0}));
    }
    {
        Tensor a{2, 1};
        Tensor b{1, 2};

        a.data_ = {1.0, 2.0};
        b.data_ = {2.0, 5.0};

        Tensor result = a.dot(b);
        check(result.data_ == std::vector<float>({2.0, 5.0, 4.0, 10.0}));
    }
    return true;
}
} // namespace test

int main()
{
    double load_time = 0.0;
    double apply_time = 0.0;

    if (!test::basics())
        return 1;

    if (!test::dense_1x1(load_time, apply_time))
        return 1;

    if (!test::dense_10x1(load_time, apply_time))
        return 1;

    if (!test::dense_2x2(load_time, apply_time))
        return 1;

    if (!test::dense_10x10(load_time, apply_time))
        return 1;

    if (!test::dense_10x10x10(load_time, apply_time))
        return 1;

    if (!test::conv_2x2(load_time, apply_time))
        return 1;

    if (!test::conv_3x3(load_time, apply_time))
        return 1;

    if (!test::conv_3x3x3(load_time, apply_time))
        return 1;

    if (!test::elu_10(load_time, apply_time))
        return 1;

    if (!test::relu_10(load_time, apply_time))
        return 1;

    if (!test::dense_relu_10(load_time, apply_time))
        return 1;

    if (!test::dense_tanh_10(load_time, apply_time))
        return 1;

    if (!test::conv_softplus_2x2(load_time, apply_time))
        return 1;

    if (!test::conv_hard_sigmoid_2x2(load_time, apply_time))
        return 1;

    if (!test::conv_sigmoid_2x2(load_time, apply_time))
        return 1;

    if (!test::maxpool2d_1x1(load_time, apply_time))
        return 1;

    if (!test::maxpool2d_2x2(load_time, apply_time))
        return 1;

    if (!test::maxpool2d_3x2x2(load_time, apply_time))
        return 1;

    if (!test::maxpool2d_3x3x3(load_time, apply_time))
        return 1;

    if (!test::lstm_simple_7x20(load_time, apply_time))
        return 1;

    if (!test::lstm_simple_stacked_16x9(load_time, apply_time))
        return 1;

    if (!test::lstm_stacked_64x83(load_time, apply_time))
        return 1;

    if (!test::embedding_64(load_time, apply_time))
        return 1;

    // Run benchmark 5 times and report duration.
    double total_load_time = 0.0;
    double total_apply_time = 0.0;

    for (int i = 0; i < 5; ++i) {
        if (!test::benchmark(load_time, apply_time))
            return 1;

        total_load_time += load_time;
        total_apply_time += apply_time;
    }
    printf("Benchmark network loads in %fs\n", total_load_time / 5);
    printf("Benchmark network runs in %fs\n", total_apply_time / 5);

    return 0;
}
