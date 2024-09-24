#include <iostream>
#include <complex>
#include <random>
#include "Matrix.hpp"
#include "Layer.hpp"
#include "NeuralNetwork.hpp"

std::complex<double> random_complex()
{
    std::random_device rd;                               // Obtain a random number from hardware
    std::mt19937 eng(rd());                              // Seed the generator
    std::uniform_real_distribution<double> distr(-1, 1); // Define the range
    double real_part = distr(eng);
    double imag_part = distr(eng);

    return std::complex<double>(real_part, imag_part);
}

int main()
{
    // Define the neural network
    NeuralNetwork<std::complex<double>> neural_net;

    // Add layers to the network
    NeuralNetworkLayer<std::complex<double>> layer1(3, 5);
    layer1.initializeWeights(random_complex);
    NeuralNetworkLayer<std::complex<double>> layer2(5, 2);
    layer2.initializeWeights(random_complex);

    neural_net.addLayer(layer1);
    neural_net.addLayer(layer2);

    // Example input vector (1x3) of complex numbers
    Matrix<std::complex<double>> input_vector(1, 3);
    input_vector[0][0] = std::complex<double>(1.0, -0.5);
    input_vector[0][1] = std::complex<double>(-2.0, 1.5);
    input_vector[0][2] = std::complex<double>(0.5, 0.5);

    // Forward pass
    Matrix<std::complex<double>> output = neural_net.forward(input_vector);

    // Print output
    std::cout << "Output after forward pass:" << std::endl;
    output.print();

    return 0;
}
