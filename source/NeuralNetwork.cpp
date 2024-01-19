#include "NeuralNetwork.hpp"
#include <cmath>
#include <random>

double SigmoidActivation(const double &x)
{
  return 1 / (1 + std::exp(-x));
}

double SigmoidActivationDerivative(const double &x)
{
  return SigmoidActivation(x) * (1 - SigmoidActivation(x));
}

double SigmoidActivationDerivativeForOutput(const double &y)
{
  return y * (1 - y);
}

double TanhActivation(const double &x)
{
  return std::tanh(x);
  // return 2 / (1 + std::exp(-2 * x)) - 1;
}

double TanhActivationDerivative(const double &x)
{
  const double factor = TanhActivation(x);
  return 1 - factor * factor;
}

double TanhActivationDerivativeForOutput(const double &y)
{
  return 1 - y * y;
}

double ReLUActivation(const double &x)
{
  return x < 0 ? 0 : x;
}

double ReLUActivationDerivative(const double &x)
{
  return x < 0 ? 0 : 1;
}

double ParametricReLUActivation(const double &x, const double &a)
{
  return x < 0 ? 0 : a * x;
}

double ParametricReLUActivationDerivative(const double &x, const double &a)
{
  return x < 0 ? a : 1;
}

double ELUActivation(const double &x, const double &a)
{
  return x < 0 ? 0 : a * (std::exp(x) - 1);
}

double ELUActivationDerivative(const double &x, const double &a)
{
  return x < 0 ? ELUActivation(x, a) + a : 1;
}

double ELUActivationDerivativeForOutput(const double &x, const double &y, const double &a)
{
  return x < 0 ? y + a : 1;
}

double SoftPlusActivation(const double &x)
{
  return std::log(1 + std::exp(x));
}

double SoftPlusActivationDerivative(const double &x)
{
  return 1 / (1 + std::exp(-x));
}

double Cost(
    const double &x,
    const double &y)
{
  double error = y - x;
  return error * error;
}

double CostDerivativeRespectToX(
    const double &x,
    const double &y)
{
  return -2 * (y - x);
}

NeuralNetwork::NeuralNetwork(const std::vector<uint64_t> &topology) : m_Topology(topology)
{
  // reserve for layers
  m_Biases.reserve(topology.size() - 1);
  m_Weights.reserve(topology.size() - 1);
  m_CachedDeltas.reserve(topology.size() - 1);
  m_CachedSums.reserve(topology.size() - 1);
  m_CachedOutputs.reserve(topology.size() - 1);

  for (uint64_t i = 1; i < topology.size(); i++)
  {
    // IMPORTANT! this opeartion will used throughout
    // topology is inputed as following {input layer, hidden layer 1, hidden layer 2, hidden layer n, output layer}
    // we don't need to create memory for the input layer as it will be passed by the user
    // that's why the real index with relative to topology index is topology index - 1
    const uint64_t l = i - 1;

    // reserve for neuron in layer i
    m_CachedSums.push_back(std::vector<double>());
    m_CachedSums[l].reserve(topology[i]);

    // reserve for neuron in layer i
    m_CachedOutputs.push_back(std::vector<double>());
    m_CachedOutputs[l].reserve(topology[i]);

    // reserve for neuron in layer i
    m_CachedDeltas.push_back(std::vector<double>());
    m_CachedDeltas[l].reserve(topology[i]);

    // reserve for neuron in layer i
    m_Biases.push_back(std::vector<double>());
    m_Biases[l].reserve(topology[i]);

    // reserve for neuron in layer i
    m_Weights.push_back(std::vector<std::vector<double>>());
    m_Weights[l].reserve(topology[i]);

    for (uint64_t j = 0; j < topology[i]; j++)
    {
      // initialize for neuron j in layer i
      m_CachedSums[l].push_back(0);
      m_CachedOutputs[l].push_back(0);
      m_CachedDeltas[l].push_back(0);
      m_Biases[l].push_back(0);
      // m_Biases[l].push_back(std::sqrt(1 / topology[i]) * std::rand());

      // reserve for layer l that affects neuron j
      m_Weights[l].push_back(std::vector<double>());
      m_Weights[l][j].reserve(topology[l]);

      for (uint64_t z = 0; z < topology[l]; z++)
      {
        // initialize weights
        // m_Weights[l][j].push_back((double)std::rand() / RAND_MAX * sqrt(1.0 / topology[l]));
        m_Weights[l][j].push_back(((double)std::rand() / RAND_MAX - 0.5) * 2 * sqrt(1.0 / topology[l]));
      }
    }
  }
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::CalculateOutputs(const std::vector<double> &inputData)
{
  // layer after input layer
  for (uint64_t i = 0; i < m_Topology[1]; i++)
  {
    m_CachedSums.front()[i] = m_Biases.front()[i];

    // product = output * weight (some way connected to the output)
    // calculated the products of all neurons of the input layer that is connected to neuron i and sum them
    for (uint64_t j = 0; j < m_Topology.front(); j++)
    {
      m_CachedSums.front()[i] += inputData[j] * m_Weights.front()[i][j];
    }

    // attempt to work with different activation function
    // m_CachedOutputs.front()[i] = ReLUActivation(m_CachedSums.front()[i]);
    // m_CachedOutputs.front()[i] = SigmoidActivation(m_CachedSums.front()[i]);
    m_CachedOutputs.front()[i] = TanhActivation(m_CachedSums.front()[i]);
  }

  // more hidden layer
  for (uint64_t i = 2; i < m_Topology.size() - 1; i++)
  {
    const uint64_t l = i - 1;

    for (uint64_t j = 0; j < m_Topology[i]; j++)
    {
      m_CachedSums[l][j] = m_Biases[l][j];

      // product = output * weight (some way connected to the output)
      // calculated the products of all neurons of the prev layer that is connected to neuron i and sum them
      for (uint64_t z = 0; z < m_Topology[l]; z++)
      {
        m_CachedSums[l][j] += m_CachedOutputs[l - 1][z] * m_Weights[l][j][z];
      }

      // m_CachedOutputs[l][i] = ReLUActivation(m_CachedSums[l][i]);
      // m_CachedOutputs[l][i] = SigmoidActivation(m_CachedSums[l][i]);
      m_CachedOutputs[l][j] = TanhActivation(m_CachedSums[l][j]);
    }
  }

  // output layer
  for (uint64_t i = 0; i < m_Topology.back(); i++)
  {
    const uint64_t l = m_Topology.size() - 2;

    m_CachedSums.back()[i] = m_Biases.back()[i];

    // product = output * weight (some way connected to the output)
    // calculated the products of all neurons of the prev hidden layer that is connected to neuron i and sum them
    for (uint64_t j = 0; j < m_Topology[l]; j++)
    {
      m_CachedSums.back()[i] += m_CachedOutputs[l - 1][j] * m_Weights.back()[i][j];
    }

    // attempt to work with different activation function for output layer
    // m_CachedOutputs.back()[i] = ReLUActivation(m_CachedSums.back()[i]);
    // m_CachedOutputs.back()[i] = SigmoidActivation(m_CachedSums.back()[i]);
    m_CachedOutputs.back()[i] = SigmoidActivation(m_CachedSums.back()[i]);
  }
}

void NeuralNetwork::CalculateDeltas(const std::vector<double> &expectedData)
{
  // output layer
  for (uint64_t i = 0; i < m_Topology.back(); i++)
  {
    // attempt to work with different activation function derivatiove for output layer
    m_CachedDeltas.back()[i] = SigmoidActivationDerivativeForOutput(m_CachedOutputs.back()[i]) * CostDerivativeRespectToX(m_CachedOutputs.back()[i], expectedData[i]) / m_Topology.back();
    // m_CachedDeltas.back()[i] = SoftPlusActivationDerivative(m_CachedSums.back()[i]) * CostDerivativeRespectToX(m_CachedOutputs.back()[i], expectedData[i]) / m_Topology.back();
    // m_CachedDeltas.back()[i] = SoftPlusActivationDerivativeForOutput(m_CachedOutputs.back()[i]) * CostDerivativeRespectToX(m_CachedOutputs.back()[i], expectedData[i]) / m_Topology.back();
  }

  // hidden layers
  for (uint64_t i = m_Topology.size() - 2; i > 0; i--)
  {
    const uint64_t l = i - 1;
    for (uint64_t j = 0; j < m_Topology[i]; j++)
    {
      // initialize delta to 0
      m_CachedDeltas[l][j] = 0;

      // sum all deltas of the next layer of layer i multiplied by the weights connected between neurons in the next layer with the neuron j
      for (uint64_t z = 0; z < m_Topology[i + 1]; z++)
      {
        m_CachedDeltas[l][j] += m_CachedDeltas[i][z] * m_Weights[i][z][j];
      }

      // attempt to work with different activation function derivatiove for hidden layer
      m_CachedDeltas[l][j] *= TanhActivationDerivativeForOutput(m_CachedOutputs[l][j]);
      // m_CachedDeltas[l][j] *= SigmoidActivationDerivativeForOutput(m_CachedOutputs[l][j]);
      // m_CachedDeltas[l][j] *= ReLUActivationDerivative(m_CachedSums[l][j]);
      // m_CachedDeltas[l][j] *= SoftPlusActivationDerivative(m_CachedSums[l][j]);
    }
  }
}

void NeuralNetwork::ApplyGradientDescent(const std::vector<double> &inputData, const double &learningRate)
{
  for (uint64_t i = 0; i < m_Topology[1]; i++)
  {
    // common operation to minimize redundancy
    double net = learningRate * m_CachedDeltas.front()[i];

    // change bias by net * 1
    m_Biases.front()[i] -= net;

    for (uint64_t j = 0; j < m_Topology.front(); j++)
    {
      // change weight by net * input
      m_Weights.front()[i][j] -= net * inputData[j];
    }
  }

  for (uint64_t i = 2; i < m_Topology.size(); i++)
  {
    const uint64_t l = i - 1;
    for (uint64_t j = 0; j < m_Topology[i]; j++)
    {
      // common operation to minimize redundancy
      double net = learningRate * m_CachedDeltas[l][j];

      // change bias by net * 1
      m_Biases[l][j] -= net;

      for (uint64_t z = 0; z < m_Topology[l]; z++)
      {
        // change weight by net * prev layer output
        m_Weights[l][j][z] -= net * m_CachedOutputs[l - 1][z];
      }
    }
  }
}

double NeuralNetwork::GetMSE(const std::vector<double> &expectedData)
{
  double SumSE = 0;
  for (uint64_t i = 0; i < m_Topology.back(); i++)
  {
    SumSE += Cost(m_CachedOutputs.back()[i], expectedData[i]);
  }

  return SumSE / (double)m_Topology.back();
}

void NeuralNetwork::Fit(const std::vector<double> &trainingData, const std::vector<double> &expectedData, const double &learningRate)
{
  CalculateOutputs(trainingData);
  CalculateDeltas(expectedData);
  ApplyGradientDescent(trainingData, learningRate);
}

PredictedData NeuralNetwork::Predict(const std::vector<double> &inputData)
{
  CalculateOutputs(inputData);
  PredictedData predictedData(-1, -1000);

  for (uint64_t i = 0; i < m_Topology.back(); i++)
  {
    if (m_CachedOutputs.back()[i] <= predictedData.m_NeuronValue)
    {
      continue;
    }

    predictedData.m_NeuronIndex = i;
    predictedData.m_NeuronValue = m_CachedOutputs.back()[i];
  }

  return predictedData;
}