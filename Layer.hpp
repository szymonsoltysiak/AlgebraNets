#ifndef NN_LAYER_HPP
#define NN_LAYER_HPP

#include "Matrix.hpp"
#include <functional>

template <typename T>
class NeuralNetworkLayer
{
private:
    Matrix<T> weights;
    Matrix<T> bias;
    size_t input_size;
    size_t output_size;
    std::function<T(T)> activation_func;
    std::function<T(T)> activation_derivative;

public:
    NeuralNetworkLayer(size_t input_size, size_t output_size)
        : input_size(input_size), output_size(output_size),
          weights(input_size, output_size), bias(1, output_size),
          activation_func([](T x)
                          { return x; }),
          activation_derivative([](T x)
                                { return 1; })
    {
    }

    NeuralNetworkLayer(size_t input_size, size_t output_size, std::function<T(T)> activation, std::function<T(T)> activation_deriv)
        : input_size(input_size), output_size(output_size),
          weights(input_size, output_size), bias(1, output_size),
          activation_func(activation),
          activation_derivative(activation_deriv)
    {
    }

    void initializeWeights(std::function<T()> random_func)
    {
        for (size_t i = 0; i < input_size; ++i)
        {
            for (size_t j = 0; j < output_size; ++j)
            {
                weights[i][j] = random_func();
            }
        }
    }

    Matrix<T> forward(const Matrix<T> &input_vector) const
    {
        if (input_vector.getRows() != 1 || input_vector.getCols() != input_size)
        {
            throw std::invalid_argument("Input vector dimensions do not match layer's input size.");
        }
        Matrix<T> output = input_vector * weights;
        output = output + bias;

        for (size_t i = 0; i < output.getRows(); ++i)
        {
            for (size_t j = 0; j < output.getCols(); ++j)
            {
                output[i][j] = activation_func(output[i][j]);
            }
        }

        return output;
    }

    void backward(const Matrix<T> &input_vector, const Matrix<T> &output_error, T learning_rate)
    {
        // Step 1: Compute the derivative of the activation function for each output element
        Matrix<T> activation_gradient(1, output_size);
        for (size_t i = 0; i < output_size; ++i)
        {
            activation_gradient[0][i] = activation_derivative(output_error[0][i]);
        }

        // Step 2: Multiply the error by the activation function's derivative
        Matrix<T> delta = output_error;
        for (size_t i = 0; i < delta.getRows(); ++i)
        {
            for (size_t j = 0; j < delta.getCols(); ++j)
            {
                delta[i][j] *= activation_gradient[0][j];
            }
        }

        // Step 3: Compute the gradient w.r.t. weights and biases
        Matrix<T> weight_gradient = input_vector.transpose() * delta; // Transpose input for proper dimensions
        Matrix<T> bias_gradient = delta;                              // Bias gradient is just the delta values

        // Step 4: Update the weights and biases using the computed gradients
        weights = weights - (weight_gradient * learning_rate); // Update weights
        bias = bias - (bias_gradient * learning_rate);         // Update biases
    }

    void setWeights(const Matrix<T> &new_weights)
    {
        if (new_weights.getRows() != input_size || new_weights.getCols() != output_size)
        {
            throw std::invalid_argument("New weights dimensions do not match.");
        }
        weights = new_weights;
    }

    void setBias(const Matrix<T> &new_bias)
    {
        if (new_bias.getRows() != 1 || new_bias.getCols() != output_size)
        {
            throw std::invalid_argument("New bias dimensions do not match.");
        }
        bias = new_bias;
    }

    void print() const
    {
        std::cout << "Weights:" << std::endl;
        weights.print();
        std::cout << "Bias:" << std::endl;
        bias.print();
    }
};

#endif // NN_LAYER_HPP
