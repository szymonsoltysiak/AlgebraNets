#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <stdexcept>

template <typename T>
class Matrix
{
private:
    std::vector<std::vector<T>> data;
    size_t rows;
    size_t cols;

public:
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols)
    {
        data.resize(rows, std::vector<T>(cols, T()));
    }

    std::vector<T> &operator[](size_t index)
    {
        return data[index];
    }

    const std::vector<T> &operator[](size_t index) const
    {
        return data[index];
    }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    T getByIndex(size_t row, size_t col) const
    {
        return data[row][col];
    }

    Matrix<T> operator+(const Matrix<T> &other) const
    {
        if (rows != other.rows || cols != other.cols)
        {
            throw std::invalid_argument("Matrices dimensions do not match for addition.");
        }
        Matrix<T> result(rows, cols);
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                result[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    Matrix<T> operator-(const Matrix<T> &rhs) const
    {
        if (rows != rhs.getRows() || cols != rhs.getCols())
        {
            throw std::invalid_argument("Matrices must have the same dimensions for subtraction.");
        }
        Matrix<T> result(rows, cols);
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                result[i][j] = data[i][j] - rhs.data[i][j]; // Element-wise subtraction
            }
        }
        return result;
    }

    Matrix<T> operator*(const Matrix<T> &other) const
    {
        if (cols != other.rows)
        {
            throw std::invalid_argument("Matrices dimensions do not match for multiplication.");
        }
        Matrix<T> result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < other.cols; ++j)
            {
                result[i][j] = T();
                for (size_t k = 0; k < cols; ++k)
                {
                    result[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    Matrix<T> operator*(const T &scalar) const
    {
        Matrix<T> result(rows, cols);
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                result[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }

    friend Matrix<T> operator*(const T &scalar, const Matrix<T> &matrix)
    {
        return matrix * scalar;
    }

    Matrix<T> transpose() const
    {
        Matrix<T> result(cols, rows);
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                result[j][i] = data[i][j];
            }
        }
        return result;
    }

    void print() const
    {
        for (const auto &row : data)
        {
            for (const auto &element : row)
            {
                std::cout << element << " ";
            }
            std::cout << std::endl;
        }
    }
};

#endif // MATRIX_HPP
