#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if __has_include(<librealsense2/rs.hpp>)
#include <librealsense2/rs.hpp>
#define WBCD_HAVE_REALSENSE 1
#else
#define WBCD_HAVE_REALSENSE 0
#endif

#include "network_helper.hpp"

namespace {

// Tencent cloud layout: one 1280x720 H.264 stream, top row ZED stereo, bottom row D405 hands.
constexpr int kOutputWidth = 1280;
constexpr int kOutputHeight = 720;
constexpr int kOutputFps = 30;
constexpr int kDefaultBitrate = 8000000;
constexpr int kEyeWidth = 640;
constexpr int kEyeHeight = 360;
constexpr int kD405Width = 640;
constexpr int kD405Height = 480;
constexpr int kD405Fps = 30;

struct CameraRequestData {
  int width;
  int height;
  int fps;
  int bitrate;
  int enableMvHevc;
  int renderMode;
  int port;
  std::string camera;
  std::string ip;

  CameraRequestData()
      : width(0), height(0), fps(0), bitrate(0), enableMvHevc(0), renderMode(0),
        port(0) {}
};

struct NetworkDataProtocol {
  std::string command;
  int length;
  std::vector<uint8_t> data;

  NetworkDataProtocol() : length(0) {}
  NetworkDataProtocol(const std::string &cmd, const std::vector<uint8_t> &d)
      : command(cmd), length(static_cast<int>(d.size())), data(d) {}
};

int32_t readInt32(const std::vector<uint8_t> &data, size_t offset) {
  if (offset + 4 > data.size()) {
    throw std::out_of_range("Not enough data to read int32");
  }
  return static_cast<int32_t>((data[offset]) | (data[offset + 1] << 8) |
                              (data[offset + 2] << 16) |
                              (data[offset + 3] << 24));
}

std::string readCompactString(const std::vector<uint8_t> &data, size_t &offset) {
  if (offset >= data.size()) {
    throw std::out_of_range("Not enough data to read string length");
  }
  uint8_t length = data[offset++];
  if (length == 0) {
    return std::string();
  }
  if (offset + length > data.size()) {
    throw std::out_of_range("Not enough data to read string content");
  }
  std::string result(reinterpret_cast<const char *>(&data[offset]), length);
  offset += length;
  return result;
}

class CameraRequestDeserializer {
public:
  // Match Unity's CameraRequestSerializer binary wire format.
  static CameraRequestData deserialize(const std::vector<uint8_t> &data) {
    if (data.size() < 10) {
      throw std::invalid_argument("Data is too small for valid camera request");
    }

    size_t offset = 0;
    if (data[offset] != 0xCA || data[offset + 1] != 0xFE) {
      throw std::invalid_argument("Invalid magic bytes");
    }
    offset += 2;

    uint8_t version = data[offset++];
    if (version != 1) {
      throw std::invalid_argument("Unsupported protocol version");
    }

    if (offset + 28 > data.size()) {
      throw std::invalid_argument("Data too small for integer fields");
    }

    CameraRequestData result;
    result.width = readInt32(data, offset);
    result.height = readInt32(data, offset + 4);
    result.fps = readInt32(data, offset + 8);
    result.bitrate = readInt32(data, offset + 12);
    result.enableMvHevc = readInt32(data, offset + 16);
    result.renderMode = readInt32(data, offset + 20);
    result.port = readInt32(data, offset + 24);
    offset += 28;

    result.camera = readCompactString(data, offset);
    result.ip = readCompactString(data, offset);
    return result;
  }
};

class NetworkDataProtocolDeserializer {
public:
  // Decode XRoboToolkit's command envelope, e.g. OPEN_CAMERA / CLOSE_CAMERA.
  static NetworkDataProtocol deserialize(const std::vector<uint8_t> &buffer) {
    if (buffer.size() < 8) {
      throw std::invalid_argument("Buffer too small for valid protocol data");
    }

    size_t offset = 0;
    int32_t commandLength = readInt32(buffer, offset);
    offset += 4;
    if (commandLength < 0 || offset + commandLength > buffer.size()) {
      throw std::invalid_argument("Invalid command length");
    }

    std::string command;
    if (commandLength > 0) {
      command = std::string(reinterpret_cast<const char *>(&buffer[offset]),
                            commandLength);
      size_t nullPos = command.find('\0');
      if (nullPos != std::string::npos) {
        command = command.substr(0, nullPos);
      }
    }
    offset += commandLength;

    if (offset + 4 > buffer.size()) {
      throw std::invalid_argument("Buffer too small for data length");
    }

    int32_t dataLength = readInt32(buffer, offset);
    offset += 4;
    if (dataLength < 0 || offset + dataLength > buffer.size()) {
      throw std::invalid_argument("Invalid data length");
    }

    std::vector<uint8_t> data;
    if (dataLength > 0) {
      data.assign(buffer.begin() + offset, buffer.begin() + offset + dataLength);
    }

    return NetworkDataProtocol(command, data);
  }
};

template <typename T, typename... Args>
std::unique_ptr<T> make_unique_helper(Args &&...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

class D405Grabber {
public:
  D405Grabber(const std::string &name, const std::string &serial)
      : name_(name), serial_(serial) {}

  ~D405Grabber() { stop(); }

  void start() {
    running_.store(true);
    worker_ = std::thread(&D405Grabber::run, this);
  }

  void stop() {
    running_.store(false);
    if (worker_.joinable()) {
      worker_.join();
    }
    try {
      pipeline_.stop();
    } catch (...) {
    }
  }

  cv::Mat latest() const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return latest_bgr_.empty() ? cv::Mat() : latest_bgr_.clone();
  }

  const std::string &serial() const { return serial_; }

private:
#if WBCD_HAVE_REALSENSE
  // Start the D405 color stream. BGR8 is preferred to avoid a conversion.
  bool startPipeline(rs2_format format) {
    rs2::config config;
    if (!serial_.empty()) {
      config.enable_device(serial_);
    }
    config.enable_stream(RS2_STREAM_COLOR, kD405Width, kD405Height, format,
                         kD405Fps);
    profile_ = pipeline_.start(config);
    color_format_ = format;
    return true;
  }

  void run() {
    try {
      try {
        startPipeline(RS2_FORMAT_BGR8);
      } catch (const rs2::error &e) {
        std::cerr << "[" << name_
                  << "] BGR8 stream failed, retrying RGB8: " << e.what()
                  << std::endl;
        startPipeline(RS2_FORMAT_RGB8);
      }
      std::cout << "[" << name_ << "] D405 started"
                << (serial_.empty() ? "" : " serial=" + serial_) << std::endl;

      while (running_.load()) {
        rs2::frameset frames;
        if (!pipeline_.poll_for_frames(&frames)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
          continue;
        }

        rs2::video_frame color = frames.get_color_frame();
        if (!color) {
          continue;
        }

        cv::Mat frame(color.get_height(), color.get_width(), CV_8UC3,
                      const_cast<void *>(color.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat bgr;
        if (color_format_ == RS2_FORMAT_RGB8) {
          cv::cvtColor(frame, bgr, cv::COLOR_RGB2BGR);
        } else {
          bgr = frame.clone();
        }

        {
          std::lock_guard<std::mutex> lock(frame_mutex_);
          latest_bgr_ = bgr.clone();
        }
      }
    } catch (const rs2::error &e) {
      std::cerr << "[" << name_ << "] RealSense error: " << e.what()
                << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "[" << name_ << "] D405 error: " << e.what() << std::endl;
    }
  }
#else
  void run() {
    std::cerr << "[" << name_
              << "] librealsense2 headers were not available at build time; "
                 "showing placeholder frame"
              << std::endl;
    while (running_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
#endif

  std::string name_;
  std::string serial_;
  mutable std::mutex frame_mutex_;
  cv::Mat latest_bgr_;
  std::atomic<bool> running_{false};
  std::thread worker_;
#if WBCD_HAVE_REALSENSE
  rs2::pipeline pipeline_;
  rs2::pipeline_profile profile_;
  rs2_format color_format_ = RS2_FORMAT_BGR8;
#endif
};

CameraRequestData current_camera_config;
std::atomic<bool> stop_requested{false};
std::atomic<bool> streaming_active{false};
std::atomic<bool> encoding_enabled{false};
std::atomic<bool> send_enabled{false};
std::atomic<bool> preview_enabled{false};

std::unique_ptr<std::thread> listen_thread;
std::unique_ptr<std::thread> streaming_thread;
std::mutex config_mutex;
std::condition_variable streaming_cv;
std::mutex streaming_mutex;

std::unique_ptr<TCPClient> sender_ptr;
std::unique_ptr<TCPServer> server_ptr;
std::string send_to_server;
int send_to_port = 0;

std::string left_d405_serial;
std::string right_d405_serial;
int requested_bitrate = kDefaultBitrate;

bool initialize_sender() {
  int retry = 10;
  while (retry > 0 && !sender_ptr && !stop_requested.load()) {
    try {
      sender_ptr = make_unique_helper<TCPClient>(send_to_server, send_to_port);
      std::cout << "Attempting to connect to " << send_to_server << ":"
                << send_to_port << std::endl;
      sender_ptr->connect();
      return true;
    } catch (const TCPException &e) {
      std::cerr << "Failed to connect to server: " << e.what() << std::endl;
      sender_ptr = nullptr;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    retry--;
  }
  return false;
}

inline cv::Mat slMat2cvMat(sl::Mat &input) {
  int cv_type = -1;
  switch (input.getDataType()) {
  case sl::MAT_TYPE::F32_C1:
    cv_type = CV_32FC1;
    break;
  case sl::MAT_TYPE::F32_C2:
    cv_type = CV_32FC2;
    break;
  case sl::MAT_TYPE::F32_C3:
    cv_type = CV_32FC3;
    break;
  case sl::MAT_TYPE::F32_C4:
    cv_type = CV_32FC4;
    break;
  case sl::MAT_TYPE::U8_C1:
    cv_type = CV_8UC1;
    break;
  case sl::MAT_TYPE::U8_C2:
    cv_type = CV_8UC2;
    break;
  case sl::MAT_TYPE::U8_C3:
    cv_type = CV_8UC3;
    break;
  case sl::MAT_TYPE::U8_C4:
    cv_type = CV_8UC4;
    break;
  default:
    break;
  }
  return cv::Mat(input.getHeight(), input.getWidth(), cv_type,
                 input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

cv::Mat fitToTile(const cv::Mat &src_bgr, const std::string &label) {
  // Center-crop each hand camera into a 16:9 tile for the lower row.
  cv::Mat tile(kEyeHeight, kEyeWidth, CV_8UC4, cv::Scalar(0, 0, 0, 255));
  if (src_bgr.empty()) {
    cv::putText(tile, label + " unavailable", cv::Point(40, kEyeHeight / 2),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(80, 80, 255, 255),
                2, cv::LINE_AA);
    return tile;
  }

  int src_w = src_bgr.cols;
  int src_h = src_bgr.rows;
  double scale = std::max(static_cast<double>(kEyeWidth) / src_w,
                          static_cast<double>(kEyeHeight) / src_h);
  int resized_w = static_cast<int>(src_w * scale + 0.5);
  int resized_h = static_cast<int>(src_h * scale + 0.5);

  cv::Mat resized;
  cv::resize(src_bgr, resized, cv::Size(resized_w, resized_h));

  int crop_x = std::max(0, (resized_w - kEyeWidth) / 2);
  int crop_y = std::max(0, (resized_h - kEyeHeight) / 2);
  cv::Mat cropped = resized(cv::Rect(crop_x, crop_y, kEyeWidth, kEyeHeight));

  cv::cvtColor(cropped, tile, cv::COLOR_BGR2BGRA);
  cv::putText(tile, label, cv::Point(24, 48), cv::FONT_HERSHEY_SIMPLEX, 1.0,
              cv::Scalar(255, 255, 255, 255), 2, cv::LINE_AA);
  return tile;
}

void copyToCanvas(const cv::Mat &src_bgra, cv::Mat &canvas, int x, int y) {
  cv::Mat dst = canvas(cv::Rect(x, y, kEyeWidth, kEyeHeight));
  if (src_bgra.cols == kEyeWidth && src_bgra.rows == kEyeHeight &&
      src_bgra.type() == CV_8UC4) {
    src_bgra.copyTo(dst);
    return;
  }

  cv::Mat resized;
  cv::resize(src_bgra, resized, cv::Size(kEyeWidth, kEyeHeight));
  if (resized.type() == CV_8UC4) {
    resized.copyTo(dst);
  } else {
    cv::cvtColor(resized, dst, cv::COLOR_BGR2BGRA);
  }
}

GstFlowReturn on_new_sample(GstAppSink *sink, gpointer) {
  // Appsink receives encoded H.264/HEVC NAL bytes; prefix length for Unity.
  GstSample *sample = gst_app_sink_pull_sample(sink);
  if (!sample) {
    return GST_FLOW_ERROR;
  }

  GstBuffer *buffer = gst_sample_get_buffer(sample);
  GstMapInfo map;
  if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    const uint8_t *data = map.data;
    gsize size = map.size;
    if (send_enabled.load() && sender_ptr && sender_ptr->isConnected() && data &&
        size > 0) {
      try {
        std::vector<uint8_t> packet(4 + size);
        packet[0] = (size >> 24) & 0xFF;
        packet[1] = (size >> 16) & 0xFF;
        packet[2] = (size >> 8) & 0xFF;
        packet[3] = (size)&0xFF;
        std::copy(data, data + size, packet.begin() + 4);
        sender_ptr->sendData(packet);
      } catch (const std::exception &e) {
        std::cerr << "Encoded frame send failed: " << e.what() << std::endl;
        streaming_active.store(false);
      }
    }
    gst_buffer_unmap(buffer, &map);
  }

  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

std::string buildPipelineString(int bitrate, bool use_hevc, bool preview) {
  // appsrc accepts BGRA composite frames, Jetson hardware encoder emits H.264.
  std::string encoder = use_hevc ? "nvv4l2h265enc" : "nvv4l2h264enc";
  std::string parser = use_hevc ? "h265parse" : "h264parse";
  std::string encoded_caps = use_hevc
                                 ? "video/x-h265,stream-format=byte-stream,alignment=au"
                                 : "video/x-h264,stream-format=byte-stream,alignment=au";

  std::string pipeline =
      "appsrc name=mysource is-live=true format=time "
      "caps=video/x-raw,format=BGRA,width=" +
      std::to_string(kOutputWidth) + ",height=" + std::to_string(kOutputHeight) +
      ",framerate=" + std::to_string(kOutputFps) + "/1 ! "
      "videoconvert ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
      "tee name=t "
      "t. ! queue ! " +
      encoder +
      " maxperf-enable=1 insert-sps-pps=true idrinterval=15 bitrate=" +
      std::to_string(std::max(bitrate, kDefaultBitrate)) + " ! " + parser + " config-interval=-1 ! " + encoded_caps +
      " ! appsink name=mysink emit-signals=true sync=false ";

  if (preview) {
    pipeline +=
        "t. ! queue ! nvvidconv ! videoconvert ! autovideosink sync=false ";
  }

  return pipeline;
}

std::vector<std::string> discoverD405Serials() {
  // Serial-based assignment keeps left/right hands stable across /dev changes.
  std::vector<std::string> serials;
#if WBCD_HAVE_REALSENSE
  try {
    rs2::context ctx;
    for (auto &&dev : ctx.query_devices()) {
      std::string name = dev.get_info(RS2_CAMERA_INFO_NAME);
      std::string serial = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
      if (name.find("D405") != std::string::npos) {
        serials.push_back(serial);
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Failed to enumerate RealSense devices: " << e.what()
              << std::endl;
  }
#else
  std::cerr << "librealsense2 not available at build time; D405 auto-discovery "
               "disabled"
            << std::endl;
#endif
  return serials;
}

void startStreamingThread();
void stopStreamingThread();
void streamingThreadFunction();

void handleOpenCamera(const std::vector<uint8_t> &data) {
  std::cout << "Handling OPEN_CAMERA command" << std::endl;
  try {
    CameraRequestData cameraConfig = CameraRequestDeserializer::deserialize(data);
    std::cout << "Camera config - Width: " << cameraConfig.width
              << ", Height: " << cameraConfig.height
              << ", FPS: " << cameraConfig.fps
              << ", Bitrate: " << cameraConfig.bitrate
              << ", IP: " << cameraConfig.ip << ", Port: " << cameraConfig.port
              << ", type: " << cameraConfig.camera << std::endl;

    if (cameraConfig.camera != "ZED") {
      std::cout << "Unsupported camera type: " << cameraConfig.camera
                << ". WBCD sender expects the ZED-compatible XR path."
                << std::endl;
      return;
    }

    {
      // Keep XRoboToolkit's target IP/port, but force WBCD video geometry.
      std::lock_guard<std::mutex> lock(config_mutex);
      current_camera_config = cameraConfig;
      current_camera_config.width = kOutputWidth;
      current_camera_config.height = kOutputHeight;
      current_camera_config.fps = kOutputFps;
      current_camera_config.bitrate =
          std::max(cameraConfig.bitrate, requested_bitrate);
    }

    send_to_server = cameraConfig.ip;
    send_to_port = cameraConfig.port;
    std::cout << "Updated sender target to " << send_to_server << ":"
              << send_to_port << std::endl;
    std::cout << "WBCD output forced to " << kOutputWidth << "x"
              << kOutputHeight << "@" << kOutputFps << " bitrate "
              << std::max(cameraConfig.bitrate, requested_bitrate)
              << std::endl;

    startStreamingThread();
  } catch (const std::exception &e) {
    std::cerr << "Failed to parse camera config: " << e.what() << std::endl;
  }
}

void handleCloseCamera(const std::vector<uint8_t> &) {
  std::cout << "Handling CLOSE_CAMERA command" << std::endl;
  stopStreamingThread();
}

void onDataCallback(const std::string &command) {
  // Incoming TCP command starts with a 4-byte big-endian body length.
  std::vector<uint8_t> binaryData(command.begin(), command.end());
  if (binaryData.size() < 4) {
    std::cerr << "Data too small to contain length header" << std::endl;
    return;
  }

  uint32_t bodyLength = (static_cast<uint32_t>(binaryData[0]) << 24) |
                        (static_cast<uint32_t>(binaryData[1]) << 16) |
                        (static_cast<uint32_t>(binaryData[2]) << 8) |
                        static_cast<uint32_t>(binaryData[3]);
  if (4 + bodyLength > binaryData.size()) {
    std::cerr << "Data too small for declared body length" << std::endl;
    return;
  }

  std::vector<uint8_t> protocolData(binaryData.begin() + 4,
                                    binaryData.begin() + 4 + bodyLength);

  try {
    NetworkDataProtocol protocol =
        NetworkDataProtocolDeserializer::deserialize(protocolData);
    std::cout << "Received protocol command: '" << protocol.command << "'"
              << std::endl;

    if (protocol.command == "OPEN_CAMERA") {
      handleOpenCamera(protocol.data);
    } else if (protocol.command == "CLOSE_CAMERA") {
      handleCloseCamera(protocol.data);
    } else {
      std::cout << "Unknown protocol command: " << protocol.command
                << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "Failed to parse NetworkDataProtocol: " << e.what()
              << std::endl;
  }
}

void onDisconnectCallback() {
  std::cout << "Client disconnected, stopping streaming" << std::endl;
  stopStreamingThread();
}

void listenThreadFunction(const std::string &listen_address) {
  std::cout << "Listen thread started on " << listen_address << std::endl;
  while (!stop_requested.load()) {
    try {
      server_ptr = make_unique_helper<TCPServer>(listen_address);
      server_ptr->setDataCallback(onDataCallback);
      server_ptr->setDisconnectCallback(onDisconnectCallback);
      server_ptr->start();
      std::cout << "TCPServer is listening on " << listen_address << std::endl;

      while (!stop_requested.load() && server_ptr) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      if (server_ptr) {
        server_ptr->stop();
        server_ptr = nullptr;
      }
    } catch (const std::exception &e) {
      std::cerr << "Listen thread error: " << e.what() << std::endl;
      if (!stop_requested.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
      }
    }
  }
  std::cout << "Listen thread stopped" << std::endl;
}

void startStreamingThread() {
  std::lock_guard<std::mutex> lock(streaming_mutex);
  if (streaming_thread && streaming_thread->joinable()) {
    std::cout << "Streaming thread already running" << std::endl;
    return;
  }
  streaming_active.store(true);
  streaming_thread = make_unique_helper<std::thread>(streamingThreadFunction);
  std::cout << "Started WBCD streaming thread" << std::endl;
}

void stopStreamingThread() {
  std::lock_guard<std::mutex> lock(streaming_mutex);
  streaming_active.store(false);
  encoding_enabled.store(false);
  send_enabled.store(false);

  if (sender_ptr && sender_ptr->isConnected()) {
    sender_ptr->disconnect();
  }
  sender_ptr = nullptr;

  if (streaming_thread && streaming_thread->joinable()) {
    streaming_cv.notify_all();
    streaming_thread->join();
    streaming_thread = nullptr;
    std::cout << "Stopped WBCD streaming thread" << std::endl;
  }
}

void handle_sigint(int) {
  std::cout << "\nSIGINT received. Stopping all threads..." << std::endl;
  stop_requested.store(true);
  stopStreamingThread();
  if (server_ptr) {
    server_ptr->stop();
    server_ptr = nullptr;
  }
  streaming_cv.notify_all();
}

void streamingThreadFunction() {
  std::cout << "WBCD streaming thread started" << std::endl;
  std::unique_ptr<D405Grabber> left_d405;
  std::unique_ptr<D405Grabber> right_d405;

  try {
    if (!initialize_sender()) {
      std::cerr << "Failed to initialize sender, streaming thread stopping"
                << std::endl;
      return;
    }

    encoding_enabled.store(true);
    send_enabled.store(true);

    // Prefer explicit serials; auto-discovery is only a convenience fallback.
    if (left_d405_serial.empty() || right_d405_serial.empty()) {
      std::vector<std::string> serials = discoverD405Serials();
      if (left_d405_serial.empty() && !serials.empty()) {
        left_d405_serial = serials[0];
      }
      if (right_d405_serial.empty() && serials.size() > 1) {
        right_d405_serial = serials[1];
      }
    }

    left_d405 = make_unique_helper<D405Grabber>("D405 Left", left_d405_serial);
    right_d405 =
        make_unique_helper<D405Grabber>("D405 Right", right_d405_serial);
    left_d405->start();
    right_d405->start();

    sl::Camera zed;
    sl::InitParameters init_params;
    init_params.depth_mode = sl::DEPTH_MODE::NONE;
    init_params.camera_resolution = sl::RESOLUTION::HD720;
    init_params.camera_fps = kOutputFps;

    if (zed.open(init_params) != sl::ERROR_CODE::SUCCESS) {
      std::cerr << "Failed to open ZED camera in WBCD streaming thread"
                << std::endl;
      return;
    }

    CameraRequestData config;
    {
      std::lock_guard<std::mutex> lock(config_mutex);
      config = current_camera_config;
    }
    int bitrate = std::max(config.bitrate, requested_bitrate);
    std::string pipeline_str =
        buildPipelineString(bitrate, config.enableMvHevc != 0,
                            preview_enabled.load());
    std::cout << "WBCD pipeline: " << pipeline_str << std::endl;

    GError *error = nullptr;
    GstElement *pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline) {
      std::cerr << "Failed to create WBCD pipeline: " << error->message
                << std::endl;
      g_clear_error(&error);
      zed.close();
      return;
    }

    GstElement *appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");
    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
    g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), nullptr);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    sl::Mat zed_left;
    sl::Mat zed_right;
    int frame_id = 0;
    std::cout << "Starting WBCD composite loop..." << std::endl;

    while (streaming_active.load() && !stop_requested.load()) {
      if (zed.grab() != sl::ERROR_CODE::SUCCESS) {
        continue;
      }

      zed.retrieveImage(zed_left, sl::VIEW::LEFT);
      zed.retrieveImage(zed_right, sl::VIEW::RIGHT);

      cv::Mat left_bgra = slMat2cvMat(zed_left).clone();
      cv::Mat right_bgra = slMat2cvMat(zed_right).clone();
      cv::Mat d405_left = fitToTile(left_d405->latest(), "D405 Left");
      cv::Mat d405_right = fitToTile(right_d405->latest(), "D405 Right");

      // Final 2x2 canvas: ZED left/right on top, D405 left/right below.
      cv::Mat canvas(kOutputHeight, kOutputWidth, CV_8UC4,
                     cv::Scalar(0, 0, 0, 255));
      copyToCanvas(left_bgra, canvas, 0, 0);
      copyToCanvas(right_bgra, canvas, kEyeWidth, 0);
      copyToCanvas(d405_left, canvas, 0, kEyeHeight);
      copyToCanvas(d405_right, canvas, kEyeWidth, kEyeHeight);

      if (encoding_enabled.load()) {
        GstBuffer *buffer = gst_buffer_new_allocate(
            nullptr, canvas.total() * canvas.elemSize(), nullptr);
        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_WRITE);
        memcpy(map.data, canvas.data, canvas.total() * canvas.elemSize());
        gst_buffer_unmap(buffer, &map);

        GST_BUFFER_PTS(buffer) =
            gst_util_uint64_scale(frame_id, GST_SECOND, kOutputFps);
        GST_BUFFER_DURATION(buffer) =
            gst_util_uint64_scale(1, GST_SECOND, kOutputFps);
        gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
        frame_id++;
      }
    }

    std::cout << "WBCD composite loop ended, cleaning up..." << std::endl;
    gst_app_src_end_of_stream(GST_APP_SRC(appsrc));
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsrc);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);
    zed.close();
  } catch (const std::exception &e) {
    std::cerr << "WBCD streaming thread error: " << e.what() << std::endl;
  }

  if (left_d405) {
    left_d405->stop();
  }
  if (right_d405) {
    right_d405->stop();
  }
  std::cout << "WBCD streaming thread finished" << std::endl;
}

void printHelp(const char *argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n";
  std::cout << "Options:\n";
  std::cout << "  --preview                 Enable local preview\n";
  std::cout << "  --listen ADDR             Listen for XR camera commands (IP:PORT)\n";
  std::cout << "  --send                    Send video stream directly to server\n";
  std::cout << "  --server IP               Direct-send target IP\n";
  std::cout << "  --port PORT               Direct-send target port\n";
  std::cout << "  --left-d405-serial SN     RealSense serial for left hand camera\n";
  std::cout << "  --right-d405-serial SN    RealSense serial for right hand camera\n";
  std::cout << "  --bitrate BPS             Encoder bitrate (default: 8000000)\n";
  std::cout << "  --help                    Show this help message\n";
}

} // namespace

int main(int argc, char *argv[]) {
  gst_init(&argc, &argv);
  signal(SIGINT, handle_sigint);

  bool preview_enabled_local = false;
  bool listen_enabled = false;
  bool send_enabled_mode = false;
  std::string listen_address;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--preview") {
      preview_enabled_local = true;
    } else if (arg == "--listen" && i + 1 < argc) {
      listen_enabled = true;
      listen_address = argv[++i];
    } else if (arg == "--send") {
      send_enabled_mode = true;
    } else if (arg == "--server" && i + 1 < argc) {
      send_to_server = argv[++i];
    } else if (arg == "--port" && i + 1 < argc) {
      send_to_port = std::stoi(argv[++i]);
    } else if (arg == "--left-d405-serial" && i + 1 < argc) {
      left_d405_serial = argv[++i];
    } else if (arg == "--right-d405-serial" && i + 1 < argc) {
      right_d405_serial = argv[++i];
    } else if (arg == "--bitrate" && i + 1 < argc) {
      requested_bitrate = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      printHelp(argv[0]);
      return 0;
    }
  }

  if (!listen_enabled && !send_enabled_mode) {
    std::cerr << "Error: Either --listen or --send option is required"
              << std::endl;
    printHelp(argv[0]);
    return -1;
  }

  if (send_enabled_mode && (send_to_server.empty() || send_to_port == 0)) {
    std::cerr << "Error: --send mode requires both --server and --port options"
              << std::endl;
    printHelp(argv[0]);
    return -1;
  }

  preview_enabled.store(preview_enabled_local);

  if (send_enabled_mode) {
    std::cout << "Starting direct WBCD video streaming to " << send_to_server
              << ":" << send_to_port << "..." << std::endl;
    {
      std::lock_guard<std::mutex> lock(config_mutex);
      current_camera_config.width = kOutputWidth;
      current_camera_config.height = kOutputHeight;
      current_camera_config.fps = kOutputFps;
      current_camera_config.bitrate = requested_bitrate;
      current_camera_config.enableMvHevc = 0;
      current_camera_config.renderMode = 2;
      current_camera_config.camera = "ZED";
      current_camera_config.ip = send_to_server;
      current_camera_config.port = send_to_port;
    }
    startStreamingThread();
  } else if (listen_enabled) {
    std::cout << "Starting WBCD threaded video streaming server..." << std::endl;
    listen_thread =
        make_unique_helper<std::thread>(listenThreadFunction, listen_address);
    std::cout << "Server started. Press Ctrl+C to stop." << std::endl;
  }

  while (!stop_requested.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  if (listen_thread && listen_thread->joinable()) {
    listen_thread->join();
  }

  std::cout << "Shutting down..." << std::endl;
  stopStreamingThread();
  std::cout << "All threads stopped. Exiting." << std::endl;
  return 0;
}
