#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "Layer.hpp"
#include <vector>

template <typename T>
class NeuralNetwork
{
private:
    std::vector<NeuralNetworkLayer<T>> layers;

public:
    NeuralNetwork(const std::vector<size_t> &layer_sizes,
                  std::function<T(T)> activation_func,
                  std::function<T(T)> activation_derivative)
    {
        if (layer_sizes.size() < 2)
        {
            throw std::invalid_argument("The network must have at least two layers (input and output).");
        }

        for (size_t i = 1; i < layer_sizes.size(); ++i)
        {
            layers.emplace_back(layer_sizes[i - 1], layer_sizes[i], activation_func, activation_derivative);
        }
    }

    NeuralNetwork(const std::vector<size_t> &layer_sizes)
    {
        if (layer_sizes.size() < 2)
        {
            throw std::invalid_argument("The network must have at least two layers (input and output).");
        }

        for (size_t i = 1; i < layer_sizes.size(); ++i)
        {
            layers.emplace_back(layer_sizes[i - 1], layer_sizes[i]);
        }
    }

    void initializeWeights(std::function<T()> random_func)
    {
        for (auto &layer : layers)
        {
            layer.initializeWeights(random_func);
        }
    }

    Matrix<T> forward(const Matrix<T> &input)
    {
        Matrix<T> output = input;

        for (const auto &layer : layers)
        {
            output = layer.forward(output);
        }

        return output;
    }

    void print() const
    {
        for (size_t i = 0; i < layers.size(); ++i)
        {
            std::cout << "Layer " << i + 1 << ":\n";
            layers[i].print();
        }
    }

    void setWeights(size_t layer_index, const Matrix<T> &new_weights)
    {
        if (layer_index >= layers.size())
        {
            throw std::out_of_range("Layer index out of range.");
        }
        layers[layer_index].setWeights(new_weights);
    }

    void setBias(size_t layer_index, const Matrix<T> &new_bias)
    {
        if (layer_index >= layers.size())
        {
            throw std::out_of_range("Layer index out of range.");
        }
        layers[layer_index].setBias(new_bias);
    }
};

#endif // NEURAL_NETWORK_HPP
