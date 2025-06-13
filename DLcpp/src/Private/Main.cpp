#include <iostream>
#include "../Public/Network.h"

int main() {
    using TUPLE = Network::TUPLE;

    // Define XOR training data
    TUPLE trainingData = {
        { Matrix(2,1), Matrix(1,1) },
        { Matrix(2,1), Matrix(1,1) },
        { Matrix(2,1), Matrix(1,1) },
        { Matrix(2,1), Matrix(1,1) }
    };

    // Fill input and output matrices for XOR
    trainingData[0].first(0, 0) = 0; trainingData[0].first(1, 0) = 0; trainingData[0].second(0, 0) = 0;
    trainingData[1].first(0, 0) = 0; trainingData[1].first(1, 0) = 1; trainingData[1].second(0, 0) = 1;
    trainingData[2].first(0, 0) = 1; trainingData[2].first(1, 0) = 0; trainingData[2].second(0, 0) = 1;
    trainingData[3].first(0, 0) = 1; trainingData[3].first(1, 0) = 1; trainingData[3].second(0, 0) = 0;

    // No test data, we just want to see if it learns
    TUPLE testData;

    // Create network with 2 input neurons, 2 hidden neurons, 1 output neuron
    Network net({ 2, 2, 1 });

    // Train: epochs=5000, miniBatchSize=4 (whole dataset), learning rate eta=0.5
    net.stochasticGD(trainingData, 5000, 4, 0.5f, testData);

    // Test predictions
    for (auto& [input, expected] : trainingData) {
        Matrix output = net.feedForward(input);
        std::cout << "Input: [" << input(0, 0) << ", " << input(1, 0) << "] ";
        std::cout << "Predicted: " << output(0, 0) << " ";
        std::cout << "Expected: " << expected(0, 0) << "\n";
    }

    return 0;
}

