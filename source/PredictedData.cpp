#include "PredictData.hpp"

PredictedData::PredictedData(
    uint64_t neuronIndex,
    double neuronValue) : m_NeuronIndex(neuronIndex),
                          m_NeuronValue(neuronValue) {}

PredictedData::~PredictedData() {}