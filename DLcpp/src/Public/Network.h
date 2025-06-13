#pragma once
#include <vector>

#include "Matrix.h"
//Defines a feed forward neural network with n layers
class Network {
public:
	using TUPLE = std::vector<std::pair<Matrix, Matrix>>;
	using GRADIENT = TUPLE;

private:
	size_t numLayers;
	std::vector<size_t> layerSizes;

	std::vector<Matrix> weights;
	std::vector<Matrix> biases;	

	static float sigmoid(float x);
	static float sigmoidPrime(float x);
	static Matrix costDerivative(Matrix& networkOutput, Matrix& y);

	void updateMiniBatch(TUPLE batch,float eta);
	size_t evaluate(TUPLE tests);
	GRADIENT backprop(Matrix input, Matrix output);

public:
	Network(const std::vector<size_t>& layerSizes);
	Matrix feedForward(const Matrix& activation) const;
	void stochasticGD(TUPLE tuples, size_t epochs, size_t miniBatchSize, float eta, TUPLE testData);
};
