#include "NeuralNetwork.hpp"
#include <iostream>
#include <random>
#include <complex>

std::complex<double> sigmoid(std::complex<double> z)
{
    return 1.0 / (1.0 + exp(-z));
}

std::complex<double> sigmoid_derivative(std::complex<double> z)
{
    std::complex<double> s = sigmoid(z);
    return s * (1.0 - s);
}

using UniformDist = std::uniform_real_distribution<>;

int main()
{
    std::vector<size_t> layer_sizes = {10, 10, 5, 3, 1};
    NeuralNetwork<std::complex<double>> nn(layer_sizes, sigmoid, sigmoid_derivative);

    std::random_device rd;
    std::mt19937 gen(rd());

    nn.initializeWeights([&]()
                         { return std::complex<double>(UniformDist(-1, 1)(gen), UniformDist(-1, 1)(gen)); });

    Matrix<std::complex<double>> input_vector(1, 10);
    for (size_t i = 0; i < 10; ++i)
    {
        input_vector[0][i] = std::complex<double>(UniformDist(-1, 1)(gen), UniformDist(-1, 1)(gen));
    }

    Matrix<std::complex<double>> output = nn.forward(input_vector);

    std::cout << "Network output:" << std::endl;
    output.print();

    nn.print();

    return 0;
}
