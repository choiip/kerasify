# Kerasify [![pipeline status](https://gitlab.com/arquolo/kerasify/badges/master/pipeline.svg)](https://gitlab.com/arquolo/kerasify/commits/master) ![last commit](https://img.shields.io/github/last-commit/arquolo/kerasify.svg?style=flat) [![license](https://img.shields.io/github/license/arquolo/kerasify.svg?style=flat)](https://github.com/arquolo/kerasify/blob/master/LICENSE) [![tag](https://img.shields.io/github/tag-date/arquolo/kerasify.svg?style=flat)](https://github.com/arquolo/kerasify/tags)

### Kerasify is a small library for running trained Keras models from a C++ application. 

Kerasify is a small library for running trained Keras models from a C++ application. 

Design goals:

* Compatibility with image processing Sequential networks generated by Keras using TensorFlow backend.
* CPU only, no GPU
* No external dependencies, standard library, C++14 features OK.
* Model stored on disk in binary format that can be quickly read.
* Model stored in memory in contiguous block for better cache performance.
* Unit testable, rigorous unit tests.

Currently implemented Keras layers:

* Embedding, Flatten
* Dense, Conv1D, Conv2D, LocallyConnected1D
* LSTM
* BatchNormalization, MaxPooling
* Activation: ELU, HardSigmoid, Linear, Relu, Sigmoid, SoftMax, SoftPlus, SoftSign, Tanh

Looking for more Keras/C++ libraries? Check out https://github.com/pplonski/keras2cpp/

# Example

make_model.py:

```python
import numpy as np
from keras import Sequential
from keras.layers import Dense

test_x = np.random.rand(10, 10).astype('f')
test_y = np.random.rand(10).astype('f')

model = Sequential([
    Dense(1, input_dim=10)
])

model.compile(loss='mse', optimizer='adam')
model.fit(test_x, test_y, epochs=1, verbose=False)

data = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
prediction = model.predict(data)
print(prediction)

from kerasify import export_model
export_model(model, 'example.model')
```

test.cpp:

```c++
#include "keras/model.h"

using keras::Model;
using keras::Tensor;

int main() {
    // Initialize model.
    auto model = Model::load("example.model");

    // Create a 1D Tensor on length 10 for input data.
    Tensor in{10};
    in.data_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Run prediction.
    Tensor out = model(in);
    out.print();
    return 0;
}
```

# Unit tests

To run the unit tests, generate the unit test models and then run `kerasify`:

```bash
$ python3 make_tests.py
...
$ mkdir build && cd build && cmake .. && cmake --build . && cd
...
$ ./build/kerasify
TEST dense_1x1
TEST dense_10x1
TEST dense_2x2
TEST dense_10x10
TEST dense_10x10x10
TEST conv_2x2
TEST conv_3x3
TEST conv_3x3x3
TEST elu_10
TEST benchmark
TEST benchmark
TEST benchmark
TEST benchmark
TEST benchmark
Benchmark network loads in 0.022415s
Benchmark network runs in 0.022597s
```

