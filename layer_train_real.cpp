#include <iostream>
#include <random>
#include "Layer.hpp" // Make sure this includes necessary definitions for Matrix and NeuralNetworkLayer

// Define ReLU activation function
double relu(double z)
{
    return std::max(0.0, z);
}

// Define the derivative of ReLU
double relu_derivative(double z)
{
    return (z > 0) ? 1 : 0;
}

// Define a simple function to fit
Matrix<double> target_function(Matrix<double> input)
{
    Matrix<double> output(1, 2);
    output[0][0] = input[0][0] + input[0][2]; // Adjusted for real values
    output[0][1] = input[0][1] * 0.5;         // Adjusted for real values
    return output;
}

using UniformDist = std::uniform_real_distribution<>;

int main()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    int input_size = 3;
    int output_size = 2;

    // Initialize the neural network layer with real numbers
    NeuralNetworkLayer<double> layer = NeuralNetworkLayer<double>(input_size, output_size, relu, relu_derivative);
    layer.initializeWeights([&]()
                            {
                                double stddev = sqrt(2.0 / input_size);  // Xavier initialization
                                return stddev * UniformDist(-1, 1)(gen); // Only real values
                            });

    // Training loop
    int epochs = 1000;
    double learning_rate = 0.01; // Using real double values
    double decay_rate = 1.0;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double loss = 0.0;
        for (int batch = 0; batch < 100; ++batch)
        {
            Matrix<double> input(1, 3); // 1 row, 3 columns
            input[0][0] = UniformDist(-1, 1)(gen);
            input[0][1] = UniformDist(-1, 1)(gen);
            input[0][2] = UniformDist(-1, 1)(gen);

            Matrix<double> target = target_function(input);

            // Forward pass
            Matrix<double> output = layer.forward(input);

            // Compute loss (mean squared error)
            Matrix<double> error = target - output;
            loss += std::pow(error[0][0], 2) + std::pow(error[0][1], 2); // Squared error

            // Backward pass
            layer.backward(input, output, error, learning_rate);
        }

        // Learning rate decay can be implemented here if needed
        // learning_rate *= decay_rate;

        if (epoch % 100 == 0)
        {
            std::cout << "Epoch " << epoch << ", Loss: " << loss / 100 << std::endl;
        }
    }

    layer.print(); // Ensure this prints the weights and biases correctly

    return 0;
}
