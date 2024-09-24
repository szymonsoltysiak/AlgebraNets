#include "Matrix.hpp"
#include <complex>

int main()
{
    using Complex = std::complex<double>;

    Matrix<Complex> mat1(2, 2);
    mat1[0][0] = Complex(1, 2); // 1 + 2i
    mat1[0][1] = Complex(3, 4); // 3 + 4i
    mat1[1][0] = Complex(5, 6); // 5 + 6i
    mat1[1][1] = Complex(7, 8); // 7 + 8i

    Matrix<Complex> mat2(2, 2);
    mat2[0][0] = Complex(9, 10);  // 9 + 10i
    mat2[0][1] = Complex(11, 12); // 11 + 12i
    mat2[1][0] = Complex(13, 14); // 13 + 14i
    mat2[1][1] = Complex(15, 16); // 15 + 16i

    Matrix<Complex> sum = mat1 + mat2;
    Matrix<Complex> product = mat1 * mat2;

    std::cout << "Matrix 1:\n";
    mat1.print();
    std::cout << "Matrix 2:\n";
    mat2.print();
    std::cout << "Sum:\n";
    sum.print();
    std::cout << "Product:\n";
    product.print();

    return 0;
}