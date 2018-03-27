/*
 * Copyright (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include <chrono>

namespace keras {

class Timer {
public:
    Timer() {}
    void start();
    double stop();

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

} // namespace keras
