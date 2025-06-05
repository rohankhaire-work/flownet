#ifndef FLOWNET_NODE__FLOWNET_NODE_HPP_
#define FLOWNET_NODE__FLOWNET_NODE_HPP_

#include "flownet/flownet.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class FlowNetNode : public rclcpp::Node
{
public:
  FlowNetNode();
  ~FlowNetNode();

private:
  // Params
  std::string image_topic_;
  std::string weight_file_;
  std::string camera_frame_, base_frame_;
  CAMParams cam_params_;

  // Variables
  cv::Mat curr_image_, prev_image_;
  cv_bridge::CvImagePtr init_image_ptr_;
  std::unique_ptr<FlowNet> flownet_;

  // Subscriber
  image_transport::Subscriber image_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Publishers
  image_transport::Publisher flow_img_pub_;

  void timerCallback();
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &);
  void publishImage(const image_transport::Publisher &);
};

#endif // FLOWNET_NODE__FLOWNET_NODE_HPP_
