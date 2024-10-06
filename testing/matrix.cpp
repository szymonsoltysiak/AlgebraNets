#include "../generic/Matrix.hpp"
#include <complex>

int main()
{
    using Complex = std::complex<double>;

    Matrix<Complex> mat1(2, 3);
    mat1[0][0] = Complex(1, 2);  // 1 + 2i
    mat1[0][1] = Complex(3, 4);  // 3 + 4i
    mat1[0][2] = Complex(3, 8);  // 3 + 8i
    mat1[1][0] = Complex(5, 6);  // 5 + 6i
    mat1[1][1] = Complex(7, 8);  // 7 + 8i
    mat1[1][2] = Complex(3, -1); // 3 - 1i

    Matrix<Complex> mat2(3, 2);
    mat2[0][0] = Complex(9, 10);  // 9 + 10i
    mat2[0][1] = Complex(11, 12); // 11 + 12i
    mat2[1][0] = Complex(13, 14); // 13 + 14i
    mat2[1][1] = Complex(15, 16); // 15 + 16i
    mat2[2][0] = Complex(1, 1);   // 1 + 1i
    mat2[2][1] = Complex(1, 2);   // 1 + 2i

    Matrix<Complex> sum = mat1 + mat2.transpose();
    Matrix<Complex> product = mat1 * mat2;

    std::cout << "A:\n";
    mat1.print();
    std::cout << "B:\n";
    mat2.print();
    std::cout << "A + B^T:\n";
    sum.print();
    std::cout << "A*B:\n";
    product.print();

    return 0;
}