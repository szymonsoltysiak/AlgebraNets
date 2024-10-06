#include <iostream>
#include <complex>
#include <random>
#include "../../generic/Layer.hpp"

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

    NeuralNetworkLayer<std::complex<double>> layer = NeuralNetworkLayer<std::complex<double>>(input_size, output_size);
    layer.initializeWeights([&]()
                            {
    double stddev = sqrt(2.0 / input_size);
    return std::complex<double>(stddev * UniformDist(-1, 1)(gen), stddev * UniformDist(-1, 1)(gen)); });

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

            Matrix<std::complex<double>> output = layer.forward(input);

            Matrix<std::complex<double>> error = output - target;
            loss += std::norm(error[0][0]) + std::norm(error[0][1]);

            layer.backward(input, error, learning_rate);
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
