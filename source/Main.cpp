
#include "NeuralNetwork.hpp"
#include <cmath>
#include <fstream>
#include <string>

std::ifstream &Read(std::ifstream &in, void *data, std::streamsize bytes);

template <typename T>
std::ifstream &Read(std::ifstream &in, T &data)
{
  return Read(in, &data, sizeof(T));
}

uint32_t SwapEndian(uint32_t val);

bool ReadMNISTImages(std::ifstream &file, std::vector<std::vector<double>> &data, uint64_t &rows, uint64_t &columns);
bool ReadMNISTLabels(std::ifstream &file, std::vector<uint8_t> &data);
void ShuffleMNISTData(std::vector<std::vector<double>> &imageData, std::vector<uint8_t> &labelData);

#define LEARNING_RATE 0.09

int main()
{
  std::ifstream imageFile("F:\\DATA\\mnist\\train-images.idx3-ubyte", std::ios::binary);
  if (!imageFile.is_open())
  {
    std::cout << "couldn't open image file\n";
    return 1;
  }

  std::ifstream labelFile("F:\\DATA\\mnist\\train-labels.idx1-ubyte", std::ios::binary);
  if (!labelFile.is_open())
  {
    std::cout << "couldn't open label file\n";
    return 1;
  }

  uint64_t rows;
  uint64_t columns;

  std::vector<std::vector<double>> imageData;
  if (!ReadMNISTImages(imageFile, imageData, rows, columns))
  {
    // failed
    return 1;
  }

  imageFile.close();

  std::vector<uint8_t> labelData;
  if (!ReadMNISTLabels(labelFile, labelData))
  {
    // failed
    return 1;
  }

  labelFile.close();

  // init network
  NeuralNetwork network({rows * columns, 24, 12, 10});

  for (uint64_t i = 0; i < labelData.size(); i++)
  {
    const uint8_t label = labelData[i];
    std::vector<double> expectedData(10, 0);

    // set index label to 1 (hot encoding)
    expectedData[label] = 1;

    // fit
    network.Fit(imageData[i], expectedData, LEARNING_RATE);

    // log every 1000 fit
    // if (i % 1000 == 0)
    // {
    //   std::cout << "\n-------------- NETWORK AFTER FIT " << i << " --------------\n";

    //   // double outputMax = -INFINITY;
    //   // double outputMin = INFINITY;

    //   // double deltaMax = -INFINITY;
    //   // double deltaMin = INFINITY;

    //   // double sumMax = -INFINITY;
    //   // double sumMin = INFINITY;

    //   // for (uint64_t i = 1; i < network.m_Topology.size(); i++)
    //   // {
    //   //   const uint64_t l = i - 1;
    //   //   for (uint64_t j = 0; j < network.m_Topology[i]; j++)
    //   //   {
    //   //     if (network.m_CachedOutputs[l][j] > outputMax)
    //   //     {
    //   //       outputMax = network.m_CachedOutputs[l][j];
    //   //     }

    //   //     if (network.m_CachedOutputs[l][j] < outputMin)
    //   //     {
    //   //       outputMin = network.m_CachedOutputs[l][j];
    //   //     }

    //   //     if (network.m_CachedDeltas[l][j] > deltaMax)
    //   //     {
    //   //       deltaMax = network.m_CachedDeltas[l][j];
    //   //     }

    //   //     if (network.m_CachedDeltas[l][j] < deltaMin)
    //   //     {
    //   //       deltaMin = network.m_CachedDeltas[l][j];
    //   //     }

    //   //     if (network.m_CachedSums[l][j] > sumMax)
    //   //     {
    //   //       sumMax = network.m_CachedSums[l][j];
    //   //     }

    //   //     if (network.m_CachedSums[l][j] < sumMin)
    //   //     {
    //   //       sumMin = network.m_CachedSums[l][j];
    //   //     }
    //   //   }
    //   // }

    //   // std::cout << "outputMax: " << outputMax << ", outputMin: " << outputMin << '\n';
    //   // std::cout << "deltaMax: " << deltaMax << ", deltaMin: " << deltaMin << '\n';
    //   // std::cout << "sumMax: " << sumMax << ", sumMin: " << sumMin << '\n';

    //   // double costSum = 0;
    //   // std::cout << "------- Output Layer -------\n";
    //   // for (uint64_t j = 0; j < network.m_Topology.back(); j++)
    //   // {
    //   //   double cost = Cost(network.m_CachedOutputs.back()[j], expectedData[j]);
    //   //   std::cout << "Neuron " << j << " -> Value: " << network.m_CachedOutputs.back()[j] << ", Cost: " << cost << ", Expected: " << expectedData[j] << '\n';
    //   //   costSum += cost;
    //   // }

    //   // std::cout << "MSE: " << costSum / network.m_Topology.back() << '\n';
    // }
  }

  imageData.clear();
  labelData.clear();

  imageFile.open("F:\\DATA\\mnist\\t10k-images.idx3-ubyte", std::ios::binary);
  if (!imageFile.is_open())
  {
    std::cout << "couldn't open image file\n";
    return 1;
  }

  labelFile.open("F:\\DATA\\mnist\\t10k-labels.idx1-ubyte", std::ios::binary);
  if (!labelFile.is_open())
  {
    std::cout << "couldn't open label file\n";
    return 1;
  }

  if (!ReadMNISTImages(imageFile, imageData, rows, columns))
  {
    // failed
    return 1;
  }

  imageFile.close();

  if (!ReadMNISTLabels(labelFile, labelData))
  {
    // failed
    return 1;
  }

  labelFile.close();

  uint64_t correct = 0;
  uint64_t wrong = 0;

  for (uint64_t i = 0; i < labelData.size(); i++)
  {
    const uint8_t label = labelData[i];

    // predict
    PredictedData predictedData = network.Predict(imageData[i]);

    // check if the predicted neuron index is the same with the label
    if (predictedData.m_NeuronIndex == label)
    {
      correct++;
    }
    else
    {
      wrong++;
    }
  }

  // log the amount of correct and wrong predictions
  std::cout << "correct: " << correct << ", wrong: " << wrong << ", total: " << wrong + correct << std::endl;

  return 0;
}

std::ifstream &Read(std::ifstream &in, void *data, std::streamsize bytes)
{
  char *buffer = reinterpret_cast<char *>(data);
  return static_cast<std::ifstream &>(in.read(buffer, bytes));
}

uint32_t SwapEndian(uint32_t val)
{
  val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
  return (val << 16) | (val >> 16);
}

bool ReadMNISTImages(std::ifstream &file, std::vector<std::vector<double>> &data, uint64_t &rows, uint64_t &columns)
{
  uint32_t magicNumber;
  uint32_t numberOfImages;
  uint32_t numberOfRows;
  uint32_t numberOfColumns;

  Read(file, &magicNumber, 4);
  magicNumber = SwapEndian(magicNumber);
  if (magicNumber != 2051)
  {
    std::cout << "Incorrect image file magicNumber: " << magicNumber << std::endl;
    return false;
  }

  Read(file, &numberOfImages, 4);
  numberOfImages = SwapEndian(numberOfImages);

  Read(file, &numberOfRows, 4);
  numberOfRows = SwapEndian(numberOfRows);
  rows = numberOfRows;

  Read(file, &numberOfColumns, 4);
  numberOfColumns = SwapEndian(numberOfColumns);
  columns = numberOfColumns;

  data.reserve(numberOfImages);
  for (uint64_t i = 0; i < numberOfImages; i++)
  {
    data.push_back(std::vector<double>());
    data[i].reserve(numberOfRows * numberOfColumns);

    for (uint64_t j = 0; j < numberOfRows * numberOfColumns; j++)
    {
      uint8_t pixel;
      Read(file, &pixel, 1);
      data[i].push_back((double)pixel / 255);
    }
  }

  return true;
}

bool ReadMNISTLabels(std::ifstream &file, std::vector<uint8_t> &data)
{
  uint32_t magicNumber;
  uint32_t numberOfLabels;

  Read(file, &magicNumber, 4);
  magicNumber = SwapEndian(magicNumber);
  if (magicNumber != 2049)
  {
    std::cout << "Incorrect label file magicNumber: " << magicNumber << std::endl;
    return false;
  }

  Read(file, &numberOfLabels, 4);
  numberOfLabels = SwapEndian(numberOfLabels);

  data.reserve(numberOfLabels);

  for (uint64_t i = 0; i < numberOfLabels; i++)
  {
    uint8_t pixel;
    Read(file, pixel);
    data.push_back(pixel);
  }

  return true;
}

void ShuffleMNISTData(std::vector<std::vector<double>> &imageData, std::vector<uint8_t> &labelData)
{
}