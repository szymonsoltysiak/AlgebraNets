#include <iostream>
#include <complex>
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

int main()
{

    Matrix<std::complex<double>> weights(3, 2);
    weights[0][0] = std::complex<double>(0.5, 1.0);   // 0.5 + i*1.0
    weights[0][1] = std::complex<double>(-0.5, -0.5); // -0.5 + i*-0.5
    weights[1][0] = std::complex<double>(1.0, 0.5);   // 1.0 + i*0.5
    weights[1][1] = std::complex<double>(1.5, -0.5);  // 1.5 + i*-0.5
    weights[2][0] = std::complex<double>(-0.5, 0.2);  // -0.5 + i*0.2
    weights[2][1] = std::complex<double>(0.5, -1.0);  // 0.5 + i*-1.0

    Matrix<std::complex<double>> bias(1, 2);
    bias[0][0] = std::complex<double>(0.1, 0.0);  // 0.1 + i*0.0
    bias[0][1] = std::complex<double>(-0.1, 0.2); // -0.1 + i*0.2

    Matrix<std::complex<double>> input_vector(1, 3);      // 1x3 vector
    input_vector[0][0] = std::complex<double>(1.0, 0.0);  // 1.0 + i*0.0
    input_vector[0][1] = std::complex<double>(2.0, -1.0); // 2.0 + i*-1.0
    input_vector[0][2] = std::complex<double>(-1.0, 0.5); // -1.0 + i*0.5

    NeuralNetworkLayer<std::complex<double>> layer_no_activation(3, 2);
    layer_no_activation.setWeights(weights);
    layer_no_activation.setBias(bias);
    Matrix<std::complex<double>> output_no_activation = layer_no_activation.forward(input_vector);

    std::cout << "Output without ReLu:" << std::endl;
    output_no_activation.print();

    NeuralNetworkLayer<std::complex<double>> layer_activation(3, 2, relu_complex<double>, relu_derivative_complex<double>);
    layer_activation.setWeights(weights);
    layer_activation.setBias(bias);
    Matrix<std::complex<double>> output_activation = layer_activation.forward(input_vector);
    Matrix<std::complex<double>> gradient(1, 2);
    gradient[0][0] = std::complex<double>(0.1, 0.0);  // Example gradient
    gradient[0][1] = std::complex<double>(-0.1, 0.2); // Example gradient

    std::cout << "Output with ReLu:" << std::endl;
    output_activation.print();

    std::complex<double> learning_rate = std::complex<double>(0.1, 0.0); // Example learning rate
    layer_activation.backward(input_vector, gradient, learning_rate);
    output_activation = layer_activation.forward(input_vector);

    std::cout << "Output after backrop:" << std::endl;
    output_activation.print();

    return 0;
}
