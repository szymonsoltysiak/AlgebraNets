#include <iostream>
#include <random>
#include "../../generic/Layer.hpp"

Matrix<double> target_function(Matrix<double> input)
{
    Matrix<double> output(1, 2);
    output[0][0] = input[0][0] + input[0][2];
    output[0][1] = input[0][1] * 2 + 3;
    return output;
}

using UniformDist = std::uniform_real_distribution<>;

int main()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    int input_size = 3;
    int output_size = 2;

    NeuralNetworkLayer<double> layer = NeuralNetworkLayer<double>(input_size, output_size);
    layer.initializeWeights([&]()
                            {
                                double stddev = sqrt(2.0 / input_size);  
                                return stddev * UniformDist(-1, 1)(gen); });

    int epochs = 100;
    double learning_rate = 0.1;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double loss = 0.0;
        for (int batch = 0; batch < 10; ++batch)
        {
            Matrix<double> input(1, 3);
            input[0][0] = UniformDist(-1, 1)(gen);
            input[0][1] = UniformDist(-1, 1)(gen);
            input[0][2] = UniformDist(-1, 1)(gen);

            Matrix<double> target = target_function(input);

            Matrix<double> output = layer.forward(input);

            Matrix<double> error = output - target;
            loss += std::pow(error[0][0], 2) + std::pow(error[0][1], 2);

            Matrix<double> input_error(1, input_size);
            layer.backward(input, error, learning_rate);
        }

        if (epoch % 10 == 0)
        {
            std::cout << "Epoch " << epoch << ", Loss: " << loss / 10 << std::endl;
        }
    }

    layer.print();

    return 0;
}
