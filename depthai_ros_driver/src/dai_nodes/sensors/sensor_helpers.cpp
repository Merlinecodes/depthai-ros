#include "depthai_ros_driver/dai_nodes/sensors/sensor_helpers.hpp"

#include <depthai/pipeline/node/XLinkOut.hpp>

#include "camera_info_manager/camera_info_manager.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/node/VideoEncoder.hpp"
#include "depthai_bridge/ImageConverter.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/logger.hpp"

namespace depthai_ros_driver {
namespace dai_nodes {
namespace sensor_helpers {

ImagePublisher::ImagePublisher(std::shared_ptr<rclcpp::Node> node,
                               std::shared_ptr<dai::Pipeline> pipeline,
                               const std::string& qName,
                               std::function<void(dai::Node::Input in)> linkFunc,
                               bool synced,
                               bool ipcEnabled,
                               bool lowBandwidth,
                               int lowBandwidthQuality)
    : node(node), qName(qName), ipcEnabled(ipcEnabled), synced(synced) {
    if(!synced) {
        xout = setupXout(pipeline, qName);
    }

    linkCB = linkFunc;
    if(lowBandwidth) {
        encoder = sensor_helpers::createEncoder(pipeline, lowBandwidthQuality);
        linkFunc(encoder->input);

        if(!synced) {
            encoder->bitstream.link(xout->input);
        } else {
            linkCB = [&](dai::Node::Input input) { encoder->bitstream.link(input); };
        }
    } else {
        if(!synced) {
            linkFunc(xout->input);
        }
    }
}
void ImagePublisher::setup(std::shared_ptr<dai::Device> device, const ImgConverterConfig& convConf, const ImgPublisherConfig& pubConf) {
    convConfig = convConf;
    pubConfig = pubConf;
    createImageConverter(device);
    createInfoManager(device);
    if(pubConfig.topicName.empty()) {
        throw std::runtime_error("Topic name cannot be empty!");
    }
    if(ipcEnabled) {
        rclcpp::PublisherOptions pubOptions;
        pubOptions.qos_overriding_options = rclcpp::QosOverridingOptions::with_default_policies();
        imgPub = node->create_publisher<sensor_msgs::msg::Image>(pubConfig.topicName + pubConfig.topicSuffix, rclcpp::QoS(10), pubOptions);
        infoPub = node->create_publisher<sensor_msgs::msg::CameraInfo>(pubConfig.topicName + "/camera_info", rclcpp::QoS(10), pubOptions);
    } else {
        imgPubIT = image_transport::create_camera_publisher(node.get(), pubConfig.topicName + pubConfig.topicSuffix);
    }
    if(!synced) {
        dataQ = device->getOutputQueue(getQueueName(), pubConf.maxQSize, pubConf.qBlocking);
        addQueueCB(dataQ);
    }
}

void ImagePublisher::createImageConverter(std::shared_ptr<dai::Device> device) {
    converter = std::make_shared<dai::ros::ImageConverter>(convConfig.tfPrefix, convConfig.interleaved, convConfig.getBaseDeviceTimestamp);
    converter->setUpdateRosBaseTimeOnToRosMsg(convConfig.updateROSBaseTimeOnRosMsg);
    if(convConfig.lowBandwidth) {
        converter->convertFromBitstream(convConfig.encoding);
    }
    if(convConfig.addExposureOffset) {
        converter->addExposureOffset(convConfig.expOffset);
    }
    if(convConfig.reverseSocketOrder) {
        converter->reverseStereoSocketOrder();
    }
    if(convConfig.alphaScalingEnabled) {
        converter->setAlphaScaling(convConfig.alphaScaling);
    }
    if(convConfig.outputDisparity) {
        auto calHandler = device->readCalibration();
        double baseline = calHandler.getBaselineDistance(pubConfig.leftSocket, pubConfig.rightSocket, false);
        if(convConfig.reverseSocketOrder) {
            baseline = calHandler.getBaselineDistance(pubConfig.rightSocket, pubConfig.leftSocket, false);
        }
        converter->convertDispToDepth(baseline);
    }
}

void ImagePublisher::createInfoManager(std::shared_ptr<dai::Device> device) {
    infoManager = std::make_shared<camera_info_manager::CameraInfoManager>(
        node->create_sub_node(std::string(node->get_name()) + "/" + pubConfig.daiNodeName).get(), "/" + pubConfig.daiNodeName + pubConfig.infoMgrSuffix);
    if(pubConfig.calibrationFile.empty()) {
        auto info = sensor_helpers::getCalibInfo(node->get_logger(), converter, device, pubConfig.socket, pubConfig.width, pubConfig.height);
        if(pubConfig.rectified) {
            std::fill(info.d.begin(), info.d.end(), 0.0);
            std::fill(info.k.begin(), info.k.end(), 0.0);
            info.r[0] = info.r[4] = info.r[8] = 1.0;
        }
        infoManager->setCameraInfo(info);
    } else {
        infoManager->loadCameraInfo(pubConfig.calibrationFile);
    }
};
ImagePublisher::~ImagePublisher() {
    closeQueue();
};

void ImagePublisher::closeQueue() {
    if(dataQ) dataQ->close();
}
void ImagePublisher::link(dai::Node::Input in) {
    linkCB(in);
}
std::shared_ptr<dai::DataOutputQueue> ImagePublisher::getQueue() {
    return dataQ;
}
void ImagePublisher::addQueueCB(const std::shared_ptr<dai::DataOutputQueue>& queue) {
    dataQ = queue;
    qName = queue->getName();
    cbID = dataQ->addCallback([this](const std::shared_ptr<dai::ADatatype>& data) { publish(data); });
}

std::string ImagePublisher::getQueueName() {
    return qName;
}
std::pair<sensor_msgs::msg::Image::UniquePtr, sensor_msgs::msg::CameraInfo::UniquePtr> ImagePublisher::convertData(const std::shared_ptr<dai::ADatatype> data) {
    auto img = std::dynamic_pointer_cast<dai::ImgFrame>(data);
    auto info = infoManager->getCameraInfo();
    auto rawMsg = converter->toRosMsgRawPtr(img, info);
    info.header = rawMsg.header;
    sensor_msgs::msg::CameraInfo::UniquePtr infoMsg = std::make_unique<sensor_msgs::msg::CameraInfo>(info);
    sensor_msgs::msg::Image::UniquePtr msg = std::make_unique<sensor_msgs::msg::Image>(rawMsg);
    return {std::move(msg), std::move(infoMsg)};
}

void ImagePublisher::publish(std::pair<sensor_msgs::msg::Image::UniquePtr, sensor_msgs::msg::CameraInfo::UniquePtr> data) {
    if(ipcEnabled && (!pubConfig.lazyPub || detectSubscription(imgPub, infoPub))) {
        imgPub->publish(std::move(data.first));
        infoPub->publish(std::move(data.second));
    } else {
        if(!pubConfig.lazyPub || imgPubIT.getNumSubscribers() > 0) imgPubIT.publish(*data.first, *data.second);
    }
}

void ImagePublisher::publish(const std::shared_ptr<dai::ADatatype>& data) {
    if(rclcpp::ok()) {
        auto img = std::dynamic_pointer_cast<dai::ImgFrame>(data);
        auto info = infoManager->getCameraInfo();
        auto rawMsg = converter->toRosMsgRawPtr(img, info);
        info.header = rawMsg.header;
        if(ipcEnabled && (!pubConfig.lazyPub || detectSubscription(imgPub, infoPub))) {
            sensor_msgs::msg::CameraInfo::UniquePtr infoMsg = std::make_unique<sensor_msgs::msg::CameraInfo>(info);
            sensor_msgs::msg::Image::UniquePtr msg = std::make_unique<sensor_msgs::msg::Image>(rawMsg);
            imgPub->publish(std::move(msg));
            infoPub->publish(std::move(infoMsg));
        } else {
            if(!pubConfig.lazyPub || imgPubIT.getNumSubscribers() > 0) imgPubIT.publish(rawMsg, info);
        }
    }
}

std::vector<ImageSensor> availableSensors = {{"IMX378", "1080P", {"12MP", "4K", "1080P"}, true},
                                             {"OV9282", "720P", {"800P", "720P", "400P"}, false},
                                             {"OV9782", "720P", {"800P", "720P", "400P"}, true},
                                             {"OV9281", "720P", {"800P", "720P", "400P"}, true},
                                             {"IMX214", "1080P", {"13MP", "12MP", "4K", "1080P"}, true},
                                             {"IMX412", "1080P", {"13MP", "12MP", "4K", "1080P"}, true},
                                             {"OV7750", "480P", {"480P", "400P"}, false},
                                             {"OV7251", "480P", {"480P", "400P"}, false},
                                             {"IMX477", "1080P", {"12MP", "4K", "1080P"}, true},
                                             {"IMX577", "1080P", {"12MP", "4K", "1080P"}, true},
                                             {"AR0234", "1200P", {"1200P"}, true},
                                             {"IMX582", "4K", {"48MP", "12MP", "4K"}, true},
                                             {"LCM48", "4K", {"48MP", "12MP", "4K"}, true}};
const std::unordered_map<dai::CameraBoardSocket, std::string> socketNameMap = {
    {dai::CameraBoardSocket::AUTO, "rgb"},
    {dai::CameraBoardSocket::CAM_A, "rgb"},
    {dai::CameraBoardSocket::CAM_B, "left"},
    {dai::CameraBoardSocket::CAM_C, "right"},
    {dai::CameraBoardSocket::CAM_D, "left_back"},
    {dai::CameraBoardSocket::CAM_E, "right_back"},
};
const std::unordered_map<dai::CameraBoardSocket, std::string> rsSocketNameMap = {
    {dai::CameraBoardSocket::AUTO, "color"},
    {dai::CameraBoardSocket::CAM_A, "color"},
    {dai::CameraBoardSocket::CAM_B, "infra2"},
    {dai::CameraBoardSocket::CAM_C, "infra1"},
    {dai::CameraBoardSocket::CAM_D, "infra4"},
    {dai::CameraBoardSocket::CAM_E, "infra3"},
};
const std::unordered_map<NodeNameEnum, std::string> rsNodeNameMap = {
    {NodeNameEnum::RGB, "color"},
    {NodeNameEnum::Left, "infra2"},
    {NodeNameEnum::Right, "infra1"},
    {NodeNameEnum::Stereo, "depth"},
    {NodeNameEnum::IMU, "imu"},
    {NodeNameEnum::NN, "nn"},
};

const std::unordered_map<NodeNameEnum, std::string> NodeNameMap = {
    {NodeNameEnum::RGB, "rgb"},
    {NodeNameEnum::Left, "left"},
    {NodeNameEnum::Right, "right"},
    {NodeNameEnum::Stereo, "stereo"},
    {NodeNameEnum::IMU, "imu"},
    {NodeNameEnum::NN, "nn"},
};

bool rsCompabilityMode(std::shared_ptr<rclcpp::Node> node) {
    return node->get_parameter("camera.i_rs_compat").as_bool();
}
std::string getNodeName(std::shared_ptr<rclcpp::Node> node, NodeNameEnum name) {
    if(rsCompabilityMode(node)) {
        return rsNodeNameMap.at(name);
    }
    return NodeNameMap.at(name);
}

std::string getSocketName(std::shared_ptr<rclcpp::Node> node, dai::CameraBoardSocket socket) {
    if(rsCompabilityMode(node)) {
        return rsSocketNameMap.at(socket);
    }
    return socketNameMap.at(socket);
}
const std::unordered_map<std::string, dai::MonoCameraProperties::SensorResolution> monoResolutionMap = {
    {"400P", dai::MonoCameraProperties::SensorResolution::THE_400_P},
    {"480P", dai::MonoCameraProperties::SensorResolution::THE_480_P},
    {"720P", dai::MonoCameraProperties::SensorResolution::THE_720_P},
    {"800P", dai::MonoCameraProperties::SensorResolution::THE_800_P},
    {"1200P", dai::MonoCameraProperties::SensorResolution::THE_1200_P},
};

const std::unordered_map<std::string, dai::ColorCameraProperties::SensorResolution> rgbResolutionMap = {
    {"720P", dai::ColorCameraProperties::SensorResolution::THE_720_P},
    {"1080P", dai::ColorCameraProperties::SensorResolution::THE_1080_P},
    {"4K", dai::ColorCameraProperties::SensorResolution::THE_4_K},
    {"12MP", dai::ColorCameraProperties::SensorResolution::THE_12_MP},
    {"13MP", dai::ColorCameraProperties::SensorResolution::THE_13_MP},
    {"800P", dai::ColorCameraProperties::SensorResolution::THE_800_P},
    {"1200P", dai::ColorCameraProperties::SensorResolution::THE_1200_P},
    {"5MP", dai::ColorCameraProperties::SensorResolution::THE_5_MP},
    {"4000x3000", dai::ColorCameraProperties::SensorResolution::THE_4000X3000},
    {"5312X6000", dai::ColorCameraProperties::SensorResolution::THE_5312X6000},
    {"48MP", dai::ColorCameraProperties::SensorResolution::THE_48_MP},
    {"1440X1080", dai::ColorCameraProperties::SensorResolution::THE_1440X1080}};

const std::unordered_map<std::string, dai::CameraControl::FrameSyncMode> fSyncModeMap = {
    {"OFF", dai::CameraControl::FrameSyncMode::OFF},
    {"OUTPUT", dai::CameraControl::FrameSyncMode::OUTPUT},
    {"INPUT", dai::CameraControl::FrameSyncMode::INPUT},
};
const std::unordered_map<std::string, dai::CameraImageOrientation> cameraImageOrientationMap = {
    {"NORMAL", dai::CameraImageOrientation::NORMAL},
    {"ROTATE_180_DEG", dai::CameraImageOrientation::ROTATE_180_DEG},
    {"AUTO", dai::CameraImageOrientation::AUTO},
    {"HORIZONTAL_MIRROR", dai::CameraImageOrientation::HORIZONTAL_MIRROR},
    {"VERTICAL_FLIP", dai::CameraImageOrientation::VERTICAL_FLIP},
};

void basicCameraPub(const std::string& /*name*/,
                    const std::shared_ptr<dai::ADatatype>& data,
                    dai::ros::ImageConverter& converter,
                    image_transport::CameraPublisher& pub,
                    std::shared_ptr<camera_info_manager::CameraInfoManager> infoManager) {
    if(rclcpp::ok() && (pub.getNumSubscribers() > 0)) {
        auto img = std::dynamic_pointer_cast<dai::ImgFrame>(data);
        auto info = infoManager->getCameraInfo();
        auto rawMsg = converter.toRosMsgRawPtr(img);
        info.header = rawMsg.header;
        pub.publish(rawMsg, info);
    }
}

sensor_msgs::msg::CameraInfo getCalibInfo(const rclcpp::Logger& logger,
                                          std::shared_ptr<dai::ros::ImageConverter> converter,
                                          std::shared_ptr<dai::Device> device,
                                          dai::CameraBoardSocket socket,
                                          int width,
                                          int height) {
    sensor_msgs::msg::CameraInfo info;
    auto calibHandler = device->readCalibration();
    try {
        info = converter->calibrationToCameraInfo(calibHandler, socket, width, height);
    } catch(std::runtime_error& e) {
        RCLCPP_ERROR(logger, "No calibration for socket %d! Publishing empty camera_info.", static_cast<int>(socket));
    }
    return info;
}

std::shared_ptr<dai::node::XLinkOut> setupXout(std::shared_ptr<dai::Pipeline> pipeline, const std::string& name) {
    auto xout = pipeline->create<dai::node::XLinkOut>();
    xout->setStreamName(name);
    xout->input.setBlocking(false);
    xout->input.setWaitForMessage(false);
    xout->input.setQueueSize(0);
    return xout;
};
std::shared_ptr<dai::node::VideoEncoder> createEncoder(std::shared_ptr<dai::Pipeline> pipeline, int quality, dai::VideoEncoderProperties::Profile profile) {
    auto enc = pipeline->create<dai::node::VideoEncoder>();
    enc->setQuality(quality);
    enc->setProfile(profile);
    return enc;
}

bool detectSubscription(const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& pub,
                        const rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr& infoPub) {
    return (pub->get_subscription_count() > 0 || pub->get_intra_process_subscription_count() > 0 || infoPub->get_subscription_count() > 0
            || infoPub->get_intra_process_subscription_count() > 0);
}
}  // namespace sensor_helpers
}  // namespace dai_nodes
}  // namespace depthai_ros_driver
