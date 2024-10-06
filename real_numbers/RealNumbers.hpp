#pragma once

#include <functional>
#include <random>

double relu(double z)
{
    return std::max(0.0, z);
}

double relu_derivative(double z)
{
    return (z > 0) ? 1 : 0;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x)
{
    double s = sigmoid(x);
    return s * (1.0 - s);
}
