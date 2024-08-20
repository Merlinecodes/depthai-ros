#include <cstdio>
#include <iostream>
#include <chrono>
#include <unordered_map>
#include <functional>
#include <cmath>

#include "camera_info_manager/camera_info_manager.hpp"
#include "depthai_bridge/BridgePublisher.hpp"
#include "depthai_bridge/ImageConverter.hpp"
#include "depthai_bridge/SpatialDetectionConverter.hpp"
#include "depthai_ros_msgs/msg/spatial_detection_array.hpp"
#include "depthai-shared/common/Point3f.hpp"
#include "rclcpp/executors.hpp"
#include "rclcpp/node.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/float32.hpp"

#include "depthai/device/DataQueue.hpp"
#include "depthai/device/Device.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/node/ColorCamera.hpp"
#include "depthai/pipeline/node/MonoCamera.hpp"
#include "depthai/pipeline/node/SpatialDetectionNetwork.hpp"
#include "depthai/pipeline/node/StereoDepth.hpp"
#include "depthai/pipeline/node/XLinkOut.hpp"

const std::vector<std::string> label_map = {"car", "none"};

std::unordered_map<int, dai::SpatialImgDetection> previousDetections;

dai::Pipeline createPipeline(bool syncNN, bool subpixel, std::string nnPath, int confidence_thresh, int LRchecktresh, std::string resolution) {
    dai::Pipeline pipeline;
    dai::node::MonoCamera::Properties::SensorResolution monoResolution;
    auto colorCam = pipeline.create<dai::node::ColorCamera>();
    auto spatialDetectionNetwork = pipeline.create<dai::node::YoloSpatialDetectionNetwork>();
    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    auto stereo = pipeline.create<dai::node::StereoDepth>();

    auto xoutRgb = pipeline.create<dai::node::XLinkOut>();
    auto xoutDepth = pipeline.create<dai::node::XLinkOut>();
    auto xoutNN = pipeline.create<dai::node::XLinkOut>();

    xoutRgb->setStreamName("preview");
    xoutNN->setStreamName("detections");
    xoutDepth->setStreamName("depth");

    colorCam->setPreviewSize(416, 416);
    colorCam->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    colorCam->setInterleaved(false);
    colorCam->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);

    if (resolution == "720p") {
        monoResolution = dai::node::MonoCamera::Properties::SensorResolution::THE_720_P;
    } else if (resolution == "400p") {
        monoResolution = dai::node::MonoCamera::Properties::SensorResolution::THE_400_P;
    } else if (resolution == "800p") {
        monoResolution = dai::node::MonoCamera::Properties::SensorResolution::THE_800_P;
    } else if (resolution == "480p") {
        monoResolution = dai::node::MonoCamera::Properties::SensorResolution::THE_480_P;
    } else {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Invalid parameter. -> monoResolution: %s", resolution.c_str());
        throw std::runtime_error("Invalid mono camera resolution.");
    }

    monoLeft->setResolution(monoResolution);
    monoLeft->setBoardSocket(dai::CameraBoardSocket::LEFT);
    monoRight->setResolution(monoResolution);
    monoRight->setBoardSocket(dai::CameraBoardSocket::RIGHT);

    stereo->initialConfig.setConfidenceThreshold(confidence_thresh);
    stereo->setRectifyEdgeFillColor(0);  // black, to better see the cutout
    stereo->initialConfig.setLeftRightCheckThreshold(LRchecktresh);
    stereo->setSubpixel(subpixel);
    stereo->setDepthAlign(dai::CameraBoardSocket::RGB);

    spatialDetectionNetwork->setBlobPath(nnPath);
    spatialDetectionNetwork->setConfidenceThreshold(0.5f);
    spatialDetectionNetwork->input.setBlocking(false);
    spatialDetectionNetwork->setBoundingBoxScaleFactor(0.5);
    spatialDetectionNetwork->setDepthLowerThreshold(100);
    spatialDetectionNetwork->setDepthUpperThreshold(5000);

    spatialDetectionNetwork->setNumClasses(2);
    spatialDetectionNetwork->setCoordinateSize(4);
    spatialDetectionNetwork->setAnchors({10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326});
    spatialDetectionNetwork->setAnchorMasks({
        {"side52", {0, 1, 2}},
        {"side26", {3, 4, 5}},
        {"side13", {6, 7, 8}}
    });
    spatialDetectionNetwork->setIouThreshold(0.5f);

    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);

    colorCam->preview.link(spatialDetectionNetwork->input);
    if (syncNN)
        spatialDetectionNetwork->passthrough.link(xoutRgb->input);
    else
        colorCam->preview.link(xoutRgb->input);

    spatialDetectionNetwork->out.link(xoutNN->input);
    stereo->depth.link(spatialDetectionNetwork->inputDepth);
    spatialDetectionNetwork->passthroughDepth.link(xoutDepth->input);
    return pipeline;
}

void publishSpeedDistance(const std::vector<dai::SpatialImgDetection>& detections, 
rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr speedPublisher,
rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr distancePublisher
) {
    auto now = std::chrono::steady_clock::now();

    for (const auto& detection : detections) {
        int label = detection.label;
        auto position = detection.spatialCoordinates;
        float confidence = detection.confidence;

        if (previousDetections.find(label) != previousDetections.end()) {
            auto& prevDetection = previousDetections[label];
            auto det_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_diff = now - det_time;
            double distance = std::sqrt(std::pow(position.x - prevDetection.spatialCoordinates.x, 2) +
                                        std::pow(position.y - prevDetection.spatialCoordinates.y, 2) +
                                        std::pow(position.z - prevDetection.spatialCoordinates.z, 2)) / 1000;
            double speed = distance / time_diff.count();

            std::cout << "Label: " << label << ", Speed: " << speed << " m/s, Distance: " << distance << "m, Confidence: " << confidence << std::endl;

            std_msgs::msg::Float32 speedMsg;
            speedMsg.data = speed;
            speedPublisher->publish(speedMsg);

            std_msgs::msg::Float32 distanceMsg;
            distanceMsg.data = distance;
            distancePublisher->publish(distanceMsg);
        }
        previousDetections[label] = detection;
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("f1tenth_detection_node");

    std::string tfPrefix, nnPath;
    std::string camera_param_uri;
    bool syncNN, subpixel;
    int confidence_thresh = 200, LRchecktresh = 5;
    std::string monoResolution = "400p";

    node->declare_parameter("tf_prefix", "oak");
    node->declare_parameter("camera_param_uri", camera_param_uri);
    node->declare_parameter("sync_nn", true);
    node->declare_parameter("subpixel", true);
    node->declare_parameter("nnPath", "/home/merline/ws_yolov5/src/f1tenth_detection_cpp/resources/f1tenth_yolov5_416.blob");
    node->declare_parameter("confidence_thresh", confidence_thresh);
    node->declare_parameter("LRchecktresh", LRchecktresh);
    node->declare_parameter("monoResolution", monoResolution);

    node->get_parameter("tf_prefix", tfPrefix);
    node->get_parameter("camera_param_uri", camera_param_uri);
    node->get_parameter("sync_nn", syncNN);
    node->get_parameter("subpixel", subpixel);
    node->get_parameter("nnPath", nnPath);
    node->get_parameter("confidence_thresh", confidence_thresh);
    node->get_parameter("LRchecktresh", LRchecktresh);
    node->get_parameter("monoResolution", monoResolution);

    dai::Pipeline pipeline = createPipeline(syncNN, subpixel, nnPath, confidence_thresh, LRchecktresh, monoResolution);
    dai::Device device(pipeline);

    auto colorQueue = device.getOutputQueue("preview", 30, false);
    auto detectionQueue = device.getOutputQueue("detections", 30, false);
    auto depthQueue = device.getOutputQueue("depth", 30, false);

    dai::rosBridge::ImageConverter rgbConverter(tfPrefix + "_rgb_camera_optical_frame", false);
    auto rgbCameraInfo = rgbConverter.calibrationToCameraInfo(device.readCalibration(), dai::CameraBoardSocket::RGB, -1, -1);
    dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::ImgFrame> rgbPublish(colorQueue,
                                                                                       node,
                                                                                       std::string("color/image"),
                                                                                       std::bind(&dai::rosBridge::ImageConverter::toRosMsg,
                                                                                                 &rgbConverter,
                                                                                                 std::placeholders::_1,
                                                                                                 std::placeholders::_2),
                                                                                       30,
                                                                                       rgbCameraInfo,
                                                                                       "color");

    dai::rosBridge::SpatialDetectionConverter detConverter(tfPrefix + "_rgb_camera_optical_frame", 416, 416, false);
    dai::rosBridge::BridgePublisher<depthai_ros_msgs::msg::SpatialDetectionArray, dai::SpatialImgDetections> detectionPublish(
        detectionQueue,
        node,
        std::string("color/yolov5_Spatial_detections"),
        std::bind(&dai::rosBridge::SpatialDetectionConverter::toRosMsg, &detConverter, std::placeholders::_1, std::placeholders::_2),
        30);

    dai::rosBridge::ImageConverter depthConverter(tfPrefix + "_right_camera_optical_frame", true);
    auto rightCameraInfo = depthConverter.calibrationToCameraInfo(device.readCalibration(), dai::CameraBoardSocket::RIGHT, 640, 480);
    dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::ImgFrame> depthPublish(
        depthQueue,
        node,
        std::string("stereo/depth"),
        std::bind(&dai::rosBridge::ImageConverter::toRosMsg, &depthConverter, std::placeholders::_1, std::placeholders::_2),
        30,
        rightCameraInfo,
        "stereo");

    // New publishers for speed and distance
    auto speedPublisher = node->create_publisher<std_msgs::msg::Float32>("realtive_speed", 10);
    auto distancePublisher = node->create_publisher<std_msgs::msg::Float32>("distance", 10);

    depthPublish.addPublisherCallback();
    detectionPublish.addPublisherCallback();
    rgbPublish.addPublisherCallback();

    detectionQueue->addCallback([&speedPublisher, &distancePublisher](const std::shared_ptr<dai::ADatatype>& data) {
        auto detections = std::dynamic_pointer_cast<dai::SpatialImgDetections>(data);
        publishSpeedDistance(detections->detections, speedPublisher, distancePublisher);
    });

    rclcpp::spin(node);

    return 0;
}