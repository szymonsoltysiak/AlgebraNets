#pragma once

#include <complex>

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