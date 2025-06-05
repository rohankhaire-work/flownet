#include "flownet/flownet_node.hpp"

FlowNetNode::FlowNetNode() : Node("flownet_node")
{
  // Set parameters
  image_topic_ = declare_parameter<std::string>("image_topic", "");
  weight_file_ = declare_parameter<std::string>("weights_file", "");
  camera_frame_ = declare_parameter<std::string>("camera_frame", "");
  base_frame_ = declare_parameter<std::string>("base_frame", "");
  cam_params_.network_h = declare_parameter("flow_network_height", 192);
  cam_params_.network_w = declare_parameter("flow_network_width", 640);
  cam_params_.network_c = declare_parameter("flow_network_channels", 6);
  cam_params_.orig_h = declare_parameter("original_image_height", 480);
  cam_params_.orig_w = declare_parameter("original_image_width", 640);

  if(image_topic_.empty() || weight_file_.empty())
  {
    RCLCPP_ERROR(get_logger(), "Check if topic name or weight file is assigned");
    return;
  }
  // Image Transport for subscribing
  image_sub_ = image_transport::create_subscription(
    this, image_topic_,
    std::bind(&FlowNetNode::imageCallback, this, std::placeholders::_1), "raw");

  timer_ = this->create_wall_timer(std::chrono::milliseconds(50),
                                   std::bind(&FlowNetNode::timerCallback, this));

  flow_img_pub_ = image_transport::create_publisher(this, "/flow_image");

  // Get weight paths
  std::string share_dir = ament_index_cpp::get_package_share_directory("flownet");
  std::string depth_weight_path = share_dir + weight_file_;

  // Initialize TensorRT and depthEstimation class
  flownet_ = std::make_unique<FlowNet>(cam_params_, depth_weight_path);
}

FlowNetNode::~FlowNetNode()
{
  timer_->cancel();
  flownet_.reset();
}

void FlowNetNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  // Convert ROS2 image message to OpenCV format
  try
  {
    init_image_ptr_ = cv_bridge::toCvCopy(msg, "rgb8");

    // Check if the ptr is present
    if(!init_image_ptr_)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge::toCvCopy() returned nullptr!");
      return;
    }

    // Copy the image
    curr_image_ = init_image_ptr_->image;
  }
  catch(cv_bridge::Exception &e)
  {
    RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
    return;
  }
  // Set flag for new image
  new_image_available_ = true;
}

void FlowNetNode::timerCallback()
{
  // Check if the image and pointcloud exists
  if(!curr_image_.empty() && !prev_image_.empty() && new_image_available_)
  {
    auto start_time = std::chrono::steady_clock::now();

    // Run Monocular depth estimation
    flownet_->runInference(curr_image_, prev_image_);

    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    RCLCPP_INFO(this->get_logger(), "Inference took %ld ms", duration_ms);

    // Publsuh depth image and depth cloud
    publishImage(flow_img_pub_);
  }

  // Update prev image
  prev_image_ = curr_image_;
  new_image_available_ = false;
}

void FlowNetNode::publishImage(const image_transport::Publisher &pub)
{
  cv::Mat bbox_img = flownet_->flow_img_;
  // Convert OpenCV image to ROS2 message
  std_msgs::msg::Header header;
  header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  sensor_msgs::msg::Image::SharedPtr msg =
    cv_bridge::CvImage(header, "bgr8", bbox_img).toImageMsg();

  // Publish image
  pub.publish(*msg);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FlowNetNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
