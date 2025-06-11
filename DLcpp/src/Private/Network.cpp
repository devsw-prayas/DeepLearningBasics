#include "../Public/Network.h"

Network::Network(const std::vector<size_t>& layerSizes) : numLayers(layerSizes.size()), layerSizes(layerSizes) {
	weights.reserve(numLayers - 1);
	biases.reserve(numLayers - 1);
	for (size_t i = 0; i < numLayers - 1; ++i) {
		weights.emplace_back(layerSizes[i + 1], layerSizes[i]);
		biases.emplace_back(layerSizes[i + 1], 1);
	}
	for (size_t i = 0; i < numLayers - 1; ++i) {
		weights[i].fillRandom();
		biases[i].fillRandom();
	}
}