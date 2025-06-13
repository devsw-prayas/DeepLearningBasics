#include "../Public/Matrix.h"

#include <cassert>
#include <random>

float& Matrix::operator()(size_t row, size_t column){
    return this->data[index(row, column)];
}

void Matrix::zero(){
    std::fill(data.begin(), data.end(), 0);
}

void Matrix::ones(){
    std::fill(data.begin(), data.end(), 1);
}

void Matrix::fillRandom(float mean, float stddev){
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> dist(mean, stddev);
	for (auto& value : data) {
		value = dist(gen);
	}
}

Matrix Matrix::identity(size_t size){
    Matrix matrix(size);
    std::fill(matrix.data.begin(), matrix.data.end(), 0);
    for(size_t i = 0; i < size; i++) matrix.data[matrix.index(i, i)] = 1;
    return matrix;
}

Matrix Matrix::add(const Matrix& other) const{
    assert(this->rows == other.rows && this->columns == other.columns && "Incompatible matrices for addition");
    Matrix output(this->rows , this->columns);
    size_t max = this->rows * this->columns;
    for(size_t i = 0; i < max; i++)
        output.data[i] = this->data[i] + other.data[i];
    return output;
}   

Matrix Matrix::subtract(const Matrix& other) const{
    assert(this->rows == other.rows && this->columns == other.columns && "Incompatible matrices for addition");
    Matrix output(this->rows, this->columns);
    size_t max = this->rows * this->columns;
    for(size_t i = 0; i < max; i++)
        output.data[i] = this->data[i] - other.data[i];
    return output;
}

Matrix Matrix::scalarMultiply(float scalar) const{
    Matrix output(this->rows, this->columns);
    size_t max = this->rows * this->columns;
    for(size_t i = 0; i < max; i++)
        output.data[i] = this->data[i] * scalar;
    return output;
}

Matrix Matrix::product(const Matrix& other) const{
    assert(this->columns == other.rows && "Incompatible for dot product");
    Matrix transposed = other.transpose();
    Matrix output(this->rows, other.columns);
    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < other.columns; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < this->columns; ++k) {
                sum += this->data[index(i, k)] * transposed.data[transposed.index(j, k)];
            }
            output(i, j) = sum;
        }
    }
    return output;
}

Matrix Matrix::transform(float (*fn)(float)) const{
    Matrix output(this->rows, this->columns);
    size_t max = this->rows * this->columns;
    for(size_t i = 0; i < max; i++)
        output.data[i] = fn(this->data[i]);
    return output;
}

Matrix Matrix::transpose() const{
    Matrix output(this->columns, this->rows);
    for(size_t i = 0; i < this->rows; i++)
        for(size_t j = 0; j < this->columns; j++)
            output.data[output.index(j, i)] = this->data[index(i ,j)];
    return output;
}   

Matrix Matrix::hadamard(const Matrix& other) const{
    assert(this->rows == other.rows && this->columns == other.columns && "Incompatible matrices for hadmard");
    Matrix output(this->rows, this->columns);
    for(size_t i = 0; i < this->data.size(); i++) output.data[i] = this->data[i] * other.data[i];
    return output;
}

size_t Matrix::argmax() const{
    assert(columns == 1 && "argmax() should be called on column vectors.");

    size_t maxIndex = 0;
    float maxValue = data[0];

    for (size_t i = 1; i < rows; ++i) {
        float val = data[i];
        if (val > maxValue) {
            maxValue = val;
            maxIndex = i;
        }
    }

    return maxIndex;
}