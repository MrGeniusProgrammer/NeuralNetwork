#pragma once

#include <stdint.h>

struct PredictedData
{
  uint64_t m_NeuronIndex;
  double m_NeuronValue;

  PredictedData(
      uint64_t neuronIndex,
      double neuronValue);

  ~PredictedData();
};
