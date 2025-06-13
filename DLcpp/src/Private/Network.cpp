#include "../Public/Network.h"

#include <cmath>
#include <random>
#include <iostream>

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

float Network::sigmoid(float x) {
	return 1.f / (1.f + std::exp(-x));
}

float Network::sigmoidPrime(float x) {
	float s = sigmoid(x);
	return s * (1 - s);
}

Matrix Network::costDerivative(Matrix& networkOutput, Matrix& y) {
	return networkOutput.subtract(y);
}

Matrix Network::feedForward(const Matrix& activation) const {
	Matrix output = activation;
	for (size_t i = 0; i < numLayers - 1; ++i) {
		output = weights[i].product(output).add(biases[i]);
		output = output.transform(sigmoid);
	}
	return output;
}

void Network::stochasticGD(TUPLE tuples, size_t epochs, size_t miniBatchSize, float eta, TUPLE testData = {}) {
	size_t tests_ = testData.size();
	size_t trains_ = tuples.size();
	for (size_t i = 0; i < epochs; ++i) {
		std::random_device rd_;
		std::mt19937 random_(rd_());
		std::shuffle(tuples.begin(), tuples.end(), random_);
		std::vector<TUPLE> batches;

		for (size_t k = 0; k < trains_; k += miniBatchSize) {
			size_t currentBatchSize = std::min(miniBatchSize, trains_ - k);
			TUPLE miniBatch_;
			miniBatch_.reserve(currentBatchSize);
			for (size_t l = k; l < k + currentBatchSize; ++l)
				miniBatch_.push_back(tuples[l]);
			batches.push_back(std::move(miniBatch_));
		}

		for (auto& batch : batches) {
			this->updateMiniBatch(batch, eta);
		}
		if (tests_) {
			std::cout << "Epoch: " << i << " Accuracy: " << evaluate(testData) << " / " << tests_ << "\n";
		}
		else {
			std::cout << "Epoch: " << i << " complete" << "\n";
		}
	}
}

void Network::updateMiniBatch(TUPLE batch, float eta) {
	std::vector<Matrix> nablaB_, nablaW_;
	for (auto& bias : biases) nablaB_.emplace_back(bias.getRows(), bias.getColumns());
	for (auto& weight : weights) nablaW_.emplace_back(weight.getRows(), weight.getColumns());
	for (auto& [x, y] : batch) {
		GRADIENT result_ = backprop(x, y);
		for (size_t i = 0; i < result_.size(); ++i) {
			nablaB_[i] = nablaB_[i].add(result_[i].first);
			nablaW_[i] = nablaW_[i].add(result_[i].second);
		}
	}

	for (size_t i = 0; i < numLayers - 1; ++i) {
		weights[i] = weights[i].subtract(nablaW_[i].scalarMultiply(eta / static_cast<float>(batch.size())));
		biases[i] = biases[i].subtract(nablaB_[i].scalarMultiply(eta / static_cast<float>(batch.size())));
	}
}

std::vector<std::pair<Matrix, Matrix>> Network::backprop(Matrix x, Matrix y) {
	std::vector<Matrix> nablaB_, nablaW_;
	for (auto& bias : biases) nablaB_.emplace_back(bias.getRows(), bias.getColumns());
	for (auto& weight : weights) nablaW_.emplace_back(weight.getRows(), weight.getColumns());

	Matrix activation = x;
	std::vector<Matrix> activations = { activation }; // Start with input
	std::vector<Matrix> zs;

	// Forward pass
	for (size_t i = 0; i < numLayers - 1; ++i) {
		Matrix z = weights[i].product(activation).add(biases[i]);
		zs.push_back(std::move(z));
		activation = z.transform(sigmoid);
		activations.push_back(activation);
	}

	// Output layer delta
	Matrix delta = costDerivative(activations.back(), y).hadamard(zs.back().transform(sigmoidPrime));
	nablaB_.back() = delta;
	nablaW_.back() = delta.product(activations[numLayers - 2].transpose());

	// Backward pass
	for (int l = static_cast<int>(numLayers - 3); l >= 0; --l) {
		Matrix sp = zs[l].transform(sigmoidPrime);
		delta = weights[l + 1].transpose().product(delta).hadamard(sp);
		nablaB_[l] = delta;
		nablaW_[l] = delta.product(activations[l].transpose());
	}

	// Pack the gradient
	GRADIENT output;
	for (size_t i = 0; i < numLayers - 1; ++i)
		output.emplace_back(std::move(nablaB_[i]), std::move(nablaW_[i]));

	return output;
}

size_t Network::evaluate(TUPLE tests){
	size_t output = 0;
	for(auto& [x, y] : tests)
		output += feedForward(x).argmax() == y.argmax();
	return output;
}