/*
 * Copyright (c) 2016 Robert W. Rose
 * Copyright (c) 2018 Paul Maevskikh
 *
 * MIT License, see LICENSE file.
 */
#include "keras/io.h"
#include <fstream>

namespace keras {

class Stream::_Impl {
    std::ifstream stream_;

    friend class Stream;

public:
    _Impl(const std::string& filename) : stream_(filename, std::ios::binary) {
        stream_.exceptions();
        if (!stream_.is_open())
            throw std::runtime_error("Cannot open " + filename);
    }

    void reads(char* ptr, size_t count) {
        stream_.read(ptr, static_cast<ptrdiff_t>(count));
        if (!stream_)
            throw std::runtime_error("File read failure");
    }
};

Stream::Stream(const std::string& filename)
: impl_(std::make_unique<Stream::_Impl>(filename)) {}

Stream::~Stream() = default;

Stream& Stream::reads(char* ptr, size_t count) {
    impl_->reads(ptr, count);
    return *this;
}

} // namespace keras
