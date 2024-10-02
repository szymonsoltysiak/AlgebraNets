#include <iostream>
#include <complex>
#include <random>
#include "Layer.hpp"

template <typename T>
std::complex<T> relu_complex(std::complex<T> z)
{
    T real_part = std::max(T(0), std::real(z));
    T imag_part = std::max(T(0), std::imag(z));
    return std::complex<T>(real_part, imag_part);
}

template <typename T>
std::complex<T> relu_derivative_complex(std::complex<T> z)
{
    T real_part = (std::real(z) > 0) ? 1 : 0;
    T imag_part = (std::real(z) > 0) ? 1 : 0;
    return std::complex<T>(real_part, imag_part);
}

// Define a simple function to fit
Matrix<std::complex<double>> target_function(Matrix<std::complex<double>> input)
{
    Matrix<std::complex<double>> output(1, 2);
    output[0][0] = input[0][0] + input[0][2];
    output[0][1] = input[0][1] * std::complex<double>(0.5, 0);
    return output;
}

using UniformDist = std::uniform_real_distribution<>;

int main()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    int input_size = 3;
    int output_size = 2;

    NeuralNetworkLayer<std::complex<double>> layer = NeuralNetworkLayer<std::complex<double>>(input_size, output_size, relu_complex<double>, relu_derivative_complex<double>);
    layer.initializeWeights([&]()
                            {
    double stddev = sqrt(2.0 / input_size);
    return std::complex<double>(stddev * UniformDist(-1, 1)(gen), stddev * UniformDist(-1, 1)(gen)); });

    // Training loop
    int epochs = 1000;
    std::complex<double> learning_rate = std::complex<double>(0.1, 0.0);
    double decay_rate = 1;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double loss = 0.0;
        for (int batch = 0; batch < 100; ++batch)
        {
            Matrix<std::complex<double>> input(1, 3); // 1 row, 3 columns
            input[0][0] = std::complex<double>(UniformDist(-1, 1)(gen), UniformDist(-1, 1)(gen));
            input[0][1] = std::complex<double>(UniformDist(-1, 1)(gen), UniformDist(-1, 1)(gen));
            input[0][2] = std::complex<double>(UniformDist(-1, 1)(gen), UniformDist(-1, 1)(gen));

            Matrix<std::complex<double>> target = target_function(input);

            // Forward pass
            Matrix<std::complex<double>> output = layer.forward(input);

            // Compute loss (mean squared error)
            Matrix<std::complex<double>> error = target - output;
            loss += std::norm(error[0][0]) + std::norm(error[0][1]);

            layer.backward(input, output, error, learning_rate);
        }
        learning_rate = learning_rate * decay_rate;
        if (epoch % 100 == 0)
        {
            std::cout << "Epoch " << epoch << ", Loss: " << loss / 100 << std::endl;
        }
    }

    layer.print();

    return 0;
}
