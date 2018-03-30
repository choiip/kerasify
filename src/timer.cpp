/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#include "keras/timer.h"

namespace keras {

void Timer::start() noexcept
{
    start_ = std::chrono::high_resolution_clock::now();
}

double Timer::stop() noexcept
{
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = now - start_;
    return diff.count();
}

} // namespace keras
