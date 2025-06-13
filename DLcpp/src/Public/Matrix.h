#pragma once
#include <vector>
/*
* This implementation uses row major internally
*/
class Matrix {
	size_t rows, columns;
	std::vector<float> data;
	inline size_t index(size_t r, size_t c) const {
        return r * columns + c;
    }

public:
	Matrix(const size_t rows, const size_t columns) : rows(rows), columns(columns) {
        data.resize(rows * columns);
    }
	Matrix() : rows(2), columns(2) {
        data.resize(rows * columns);
    }

    explicit Matrix(const size_t size) : Matrix(size, size){}

	float& operator()(size_t row, size_t column);

	void zero();
	void ones();
	void fillRandom(float mean = 0.f, float stddev = 1.f);
	static Matrix identity(size_t size);

	Matrix add(const Matrix& other) const;
	Matrix subtract(const Matrix& other) const;
	Matrix scalarMultiply(float scalar) const;
	Matrix product(const Matrix& other) const;
	Matrix transform(float (*fn)(float)) const;
	Matrix hadamard(const Matrix& other) const;

	size_t argmax() const;

    Matrix transpose() const;

	inline size_t getColumns() const{
		return columns;
	}

	inline size_t getRows() const {
		return rows;
	}
};