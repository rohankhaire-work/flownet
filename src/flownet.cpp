#include "flownet/flownet.hpp"

FlowNet::FlowNet(const CAMParams &cam_params, const std::string &weight_file)
{
  // Set depth img size and detection img size
  resize_h_ = cam_params.network_h;
  resize_w_ = cam_params.network_w;
  channels_ = cam_params.network_c;
  orig_h_ = cam_params.orig_h;
  orig_w_ = cam_params.orig_w;

  // Scale w.r.t original size
  scale_x_ = static_cast<float>(resize_w_) / cam_params.orig_w;
  scale_y_ = static_cast<float>(resize_h_) / cam_params.orig_h;

  // Make colorwheel
  colorwheel_ = makeColorWheel();

  // Get plugin paths
  std::string share_dir = ament_index_cpp::get_package_share_directory("flownet");
  std::string plugin1_path = share_dir + plugin1_file_;
  std::string plugin2_path = share_dir + plugin2_file_;

  // Set up TRT
  initializeTRT(weight_file, plugin1_path, plugin2_path);

  // Allocate buffers
  cudaError_t err = cudaMallocHost(reinterpret_cast<void **>(&input_host_),
                                   1 * channels_ * resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers_[0], 1 * channels_ * resize_h_ * resize_w_ * sizeof(float));
  cudaMallocHost(reinterpret_cast<void **>(&output_host_),
                 1 * 2 * (resize_h_ / 4) * (resize_w_ / 4) * sizeof(float));
  cudaMalloc(&buffers_[1], 1 * 2 * (resize_h_ / 4) * (resize_w_ / 4) * sizeof(float));

  // Create stream
  cudaStreamCreate(&stream_);
}

FlowNet::~FlowNet()
{
  if(buffers_[0])
  {
    cudaFree(buffers_[0]);
    buffers_[0] = nullptr;
  }
  if(buffers_[1])
  {
    cudaFree(buffers_[1]);
    buffers_[1] = nullptr;
  }
  if(input_host_)
  {
    cudaFreeHost(input_host_);
    input_host_ = nullptr;
  }
  if(output_host_)
  {
    cudaFreeHost(output_host_);
    output_host_ = nullptr;
  }
  if(stream_)
  {
    cudaStreamDestroy(stream_);
  }
}

cv::Mat FlowNet::normalizeRGB(const cv::Mat &input)
{
  std::vector<cv::Mat> channels(3);
  cv::split(input, channels);

  std::vector<cv::Mat> temp_data;
  temp_data.resize(3);

  for(int i = 0; i < 3; ++i)
  {
    cv::Mat float_channel;
    channels[i].convertTo(float_channel, CV_32F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(float_channel, mean, stddev);

    // Normalize: (x - mean) / std
    temp_data[i] = (float_channel - mean[0]) / stddev[0];
  }

  // Convert to cv::Mat
  cv::Mat normalized_rgb;
  cv::vconcat(temp_data, normalized_rgb);

  return normalized_rgb;
}

std::vector<float> FlowNet::preprocessImage(const cv::Mat &image)
{
  cv::Mat resized, chw_image;

  // Resize to model input size
  cv::resize(image, resized, cv::Size(resize_w_, resize_h_));

  // Convert to float32 and CHW
  chw_image = normalizeRGB(resized);

  // Convert to Tensor
  std::vector<float> chw_tensor = imageToTensor(chw_image);

  return chw_tensor;
}

std::vector<float> FlowNet::imageToTensor(const cv::Mat &mat)
{
  std::vector<float> tensor_data;
  if(mat.isContinuous())
    tensor_data.assign((float *)mat.datastart, (float *)mat.dataend);
  else
  {
    // Convert from HWC to CHW
    if(mat.channels() == 1)
    {
      // Single-channel (grayscale)
      for(int i = 0; i < mat.rows; ++i)
      {
        const float *row_ptr = mat.ptr<float>(i);
        tensor_data.insert(tensor_data.end(), row_ptr, row_ptr + mat.cols);
      }
    }
    else
    {
      // Multi-channel (e.g., RGB = 3 channels)
      for(int c = 0; c < mat.channels(); ++c)
      {
        for(int i = 0; i < mat.rows; ++i)
        {
          for(int j = 0; j < mat.cols; ++j)
          {
            const cv::Vec<float, 3> &pixel = mat.at<cv::Vec<float, 3>>(i, j);
            tensor_data.push_back(pixel[c]);
          }
        }
      }
    }
  }
  return tensor_data;
}

void FlowNet::initializeTRT(const std::string &engine_file, const std::string &plugin1,
                            const std::string &plugin2)
{
  // --- Load plugin library ---
  void *handle1 = dlopen(plugin1.c_str(), RTLD_NOW);
  if(!handle1)
  {
    throw std::runtime_error(std::string("Failed to load plugin: ") + dlerror());
  }

  void *handle2 = dlopen(plugin2.c_str(), RTLD_NOW);
  if(!handle2)
  {
    throw std::runtime_error(std::string("Failed to load plugin2: ") + dlerror());
  }

  initLibNvInferPlugins(&gLogger, "");

  // Load TensorRT engine from file
  std::ifstream file(engine_file, std::ios::binary);
  if(!file)
  {
    throw std::runtime_error("Failed to open engine file: " + engine_file);
  }
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);

  // Create runtime and deserialize engine
  // Create TensorRT Runtime
  runtime.reset(nvinfer1::createInferRuntime(gLogger));

  // Deserialize engine
  engine.reset(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  context.reset(engine->createExecutionContext());
}

std::vector<float>
FlowNet::computeNetworkReadyInput(const cv::Mat &curr, const cv::Mat &prev)
{
  // Preprocess RGB images
  // convert to tensors
  std::vector<float> curr_tensor = preprocessImage(curr);
  std::vector<float> prev_tensor = preprocessImage(prev);

  std::vector<float> network_input;
  network_input.reserve(channels_ * resize_h_ * resize_w_);

  // Concatenate vector
  for(int i = 0; i < resize_h_ * resize_w_; ++i)
  {
    for(int c = 0; c < 3; ++c)
    {
      network_input[c * resize_h_ * resize_w_ + i] =
        prev_tensor[c * resize_h_ * resize_w_ + i];
      network_input[(3 + c) * resize_h_ * resize_w_ + i] =
        curr_tensor[c * resize_h_ * resize_w_ + i];
    }
  }

  return network_input;
}

void FlowNet::runInference(const cv::Mat &curr, const cv::Mat &prev)
{
  // Preprocess image and convert to vector
  std::vector<float> input_tensor = computeNetworkReadyInput(curr, prev);

  // Copy to host memory and then to GPU
  std::memcpy(input_host_, input_tensor.data(),
              1 * channels_ * resize_h_ * resize_w_ * sizeof(float));
  cudaMemcpyAsync(buffers_[0], input_host_,
                  1 * channels_ * resize_h_ * resize_w_ * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);

  // Set up inference buffers
  context->setInputTensorAddress("input", buffers_[0]);
  context->setOutputTensorAddress("output", buffers_[1]);

  // inference
  context->enqueueV3(stream_);

  // Copy the result back
  cudaMemcpyAsync(output_host_, buffers_[1],
                  1 * 2 * (resize_h_ / 4) * (resize_w_ / 4) * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);

  cudaStreamSynchronize(stream_);

  // Convert to cv::Mat
  int H = resize_h_ / 4;
  int W = resize_w_ / 4;
  cv::Mat u(H, W, CV_32FC1, output_host_);
  cv::Mat v(H, W, CV_32FC1, output_host_ + H * W);

  cv::Mat flow_data;
  cv::merge(std::vector<cv::Mat>{u, v}, flow_data); // CV_32FC2

  // postProcessFlow
  cv::Mat flow_processed = postProcessFlow(flow_data);

  // store the depth image
  flow_img_ = convertToFlowImg(flow_processed);
}

cv::Mat FlowNet::postProcessFlow(const cv::Mat &flow_input)
{
  // Upsample to original resolution
  cv::Mat flow_resized;
  cv::resize(flow_input, flow_resized, cv::Size(resize_w_, resize_h_), 0, 0,
             cv::INTER_LINEAR);
  flow_resized = flow_resized * div_flow_;

  cv::resize(flow_resized, flow_resized, cv::Size(orig_w_, orig_h_), 0, 0,
             cv::INTER_LINEAR);

  // Rescale flow values
  for(int y = 0; y < flow_resized.rows; ++y)
  {
    for(int x = 0; x < flow_resized.cols; ++x)
    {
      cv::Vec2f &f = flow_resized.at<cv::Vec2f>(y, x);
      f[0] *= scale_x_; // flow x
      f[1] *= scale_y_; // flow y
    }
  }

  cv::Mat u(flow_resized.rows, flow_resized.cols, CV_32FC1);
  cv::Mat v(flow_resized.rows, flow_resized.cols, CV_32FC1);

  // Split the flow into u and v components
  std::vector<cv::Mat> flow_channels(2);
  cv::split(flow_resized, flow_channels);
  u = flow_channels[0];
  v = flow_channels[1];

  // Compute magnitude (rad = sqrt(u^2 + v^2))
  cv::Mat rad;
  cv::magnitude(u, v, rad);

  // Find max magnitude
  double rad_max;
  cv::minMaxLoc(rad, nullptr, &rad_max);

  const float epsilon = 1e-5f;
  float norm_factor = 1.0f / (rad_max + epsilon);

  // Normalize u and v
  u = u * norm_factor;
  v = v * norm_factor;

  cv::Mat flow_normalized;
  cv::merge(std::vector<cv::Mat>{u, v}, flow_normalized);

  return flow_normalized;
}

cv::Mat FlowNet::convertToFlowImg(const cv::Mat &flow_uv)
{
  // Split flow into u and v
  std::vector<cv::Mat> flow_channels(2);
  cv::split(flow_uv, flow_channels);
  cv::Mat u = flow_channels[0];
  cv::Mat v = flow_channels[1];

  // 2. Compute magnitude and angle
  cv::Mat rad, angle;
  cv::magnitude(u, v, rad);
  cv::phase(u, v, angle, true); // angle in degrees

  // Normalize angle to range [0, 1] then scale to colorwheel
  angle = (180.0f - angle) / 360.0f;

  // Use the colorwheel
  int ncols = colorwheel_.size();

  // Prepare output image
  cv::Mat flow_image(u.rows, u.cols, CV_8UC3);

  for(int y = 0; y < u.rows; ++y)
  {
    for(int x = 0; x < u.cols; ++x)
    {
      float fx = u.at<float>(y, x);
      float fy = v.at<float>(y, x);
      float rad_val = std::sqrt(fx * fx + fy * fy);
      float a = std::atan2(-fy, -fx) / CV_PI; // in range [-1, 1]
      float fk = (a + 1.0f) / 2.0f * (ncols - 1);
      int k0 = static_cast<int>(std::floor(fk));
      int k1 = (k0 + 1) % ncols;
      float f = fk - k0;

      cv::Vec3b color_pixel;
      for(int i = 0; i < 3; ++i)
      {
        float col0 = colorwheel_[k0][i] / 255.0f;
        float col1 = colorwheel_[k1][i] / 255.0f;
        float col = (1.0f - f) * col0 + f * col1;

        if(rad_val <= 1.0f)
          col = 1.0f - rad_val * (1.0f - col); // increase saturation
        else
          col *= 0.75f; // desaturate

        color_pixel[2 - i] = static_cast<uchar>(255.0f * col); // BGR
      }

      flow_image.at<cv::Vec3b>(y, x) = color_pixel;
    }
  }

  return flow_image;
}

std::vector<cv::Vec3i> FlowNet::makeColorWheel()
{
  int RY = 15;
  int YG = 6;
  int GC = 4;
  int CB = 11;
  int BM = 13;
  int MR = 6;

  int ncols = RY + YG + GC + CB + BM + MR;
  std::vector<cv::Vec3i> colorwheel(ncols); // RGB

  int col = 0;

  // RY
  for(int i = 0; i < RY; ++i, ++col)
    colorwheel[col] = cv::Vec3i(255, static_cast<int>(std::floor(255.0 * i / RY)), 0);

  // YG
  for(int i = 0; i < YG; ++i, ++col)
    colorwheel[col] =
      cv::Vec3i(255 - static_cast<int>(std::floor(255.0 * i / YG)), 255, 0);

  // GC
  for(int i = 0; i < GC; ++i, ++col)
    colorwheel[col] = cv::Vec3i(0, 255, static_cast<int>(std::floor(255.0 * i / GC)));

  // CB
  for(int i = 0; i < CB; ++i, ++col)
    colorwheel[col] =
      cv::Vec3i(0, 255 - static_cast<int>(std::floor(255.0 * i / CB)), 255);

  // BM
  for(int i = 0; i < BM; ++i, ++col)
    colorwheel[col] = cv::Vec3i(static_cast<int>(std::floor(255.0 * i / BM)), 0, 255);

  // MR
  for(int i = 0; i < MR; ++i, ++col)
    colorwheel[col] =
      cv::Vec3i(255, 0, 255 - static_cast<int>(std::floor(255.0 * i / MR)));

  return colorwheel;
}
