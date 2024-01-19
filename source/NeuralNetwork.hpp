#pragma once

#include "PredictData.hpp"
#include <iostream>
#include <stdint.h>
#include <vector>

double Cost(
    const double &predicted,
    const double &actual);

class NeuralNetwork
{
private:
  void CalculateOutputs(const std::vector<double> &inputData);

  void CalculateDeltas(const std::vector<double> &expectedData);

  void ApplyGradientDescent(
      const std::vector<double> &inputData,
      const double &learningRate);

public:
  std::vector<uint64_t> m_Topology;
  std::vector<std::vector<double>> m_CachedDeltas;
  std::vector<std::vector<double>> m_CachedSums;
  std::vector<std::vector<double>> m_CachedOutputs;
  std::vector<std::vector<double>> m_Biases;
  std::vector<std::vector<std::vector<double>>> m_Weights;

  NeuralNetwork(const std::vector<uint64_t> &topology);
  ~NeuralNetwork();

  void Fit(
      const std::vector<double> &trainingData,
      const std::vector<double> &expectedData,
      const double &learningRate = 0.001);

  double GetMSE(const std::vector<double> &expectedData);

  PredictedData Predict(const std::vector<double> &inputData);
};
