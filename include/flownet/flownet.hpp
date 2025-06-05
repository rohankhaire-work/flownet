#ifndef FLOWNET__FLOWNET_HPP_
#define FLOWNET__FLOWNET_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

struct CAMParams
{
  int orig_h;
  int orig_w;
  int network_h;
  int network_w;
  int network_c;
};

class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char *msg) noexcept override
  {
    if(severity <= Severity::kINFO)
      std::cout << "[TRT] " << msg << std::endl;
  }
};

class FlowNet
{
public:
  FlowNet(const CAMParams &, const std::string &);
  ~FlowNet();

  void runInference(const cv::Mat &, const cv::Mat &);

  cv::Mat flow_img_;

private:
  int resize_h_, resize_w_, channels_;
  int orig_h_, orig_w_;
  Logger gLogger;
  std::vector<float> result_;
  float fx_, fy_, cx_, cy_;
  float div_flow_ = 20.0f;
  float scale_x_, scale_y_;
  std::vector<cv::Vec3i> colorwheel_;

  // Buffers
  void *buffers_[2];
  float *input_host_ = nullptr;
  float *output_host_ = nullptr;

  // Tensorrt
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream_;

  std::vector<float> preprocessImage(const cv::Mat &);
  std::vector<float> imageToTensor(const cv::Mat &);
  std::vector<float> computeNetworkReadyInput(const cv::Mat &, const cv::Mat &);
  void initializeTRT(const std::string &);
  cv::Mat normalizeRGB(const cv::Mat &input);
  cv::Mat postProcessFlow(const cv::Mat &);
  std::vector<cv::Vec3i> makeColorWheel();
  cv::Mat convertToFlowImg(const cv::Mat &);
};

#endif // FLOWNET__FLOWNET_HPP_
