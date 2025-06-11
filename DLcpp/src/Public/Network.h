#pragma once
#include <vector>

#include "Matrix.h"
//Defines a feed forward neural network with n layers
class Network {
	size_t numLayers;
	std::vector<size_t> layerSizes;

	std::vector<Matrix> weights;
	std::vector<Matrix> biases;

public:
	Network(const std::vector<size_t>& layerSizes);
};
