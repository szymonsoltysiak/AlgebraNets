#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include "Layer.hpp"

template <typename T>
class NeuralNetwork
{
private:
    std::vector<NeuralNetworkLayer<T>> layers;

public:
    void addLayer(const NeuralNetworkLayer<T> &layer)
    {
        layers.push_back(layer);
    }

    std::vector<NeuralNetworkLayer<T>> getLayers() const
    {
        return layers;
    }

    Matrix<T> forward(const Matrix<T> &input_vector)
    {
        Matrix<T> output = input_vector;
        for (auto &layer : layers)
        {
            output = layer.forward(output);
        }
        return output;
    }

    void printLayers() const
    {
        for (size_t i = 0; i < layers.size(); ++i)
        {
            std::cout << "Layer " << i << " Weights:" << std::endl;
            layers[i].print();
        }
    }
};

#endif