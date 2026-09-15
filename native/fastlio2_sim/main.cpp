// SPDX-License-Identifier: Apache-2.0
// Binary stdin/stdout adapter for DimOS' pinned FAST-LIO2 core.  It replaces
// only the physical Livox SDK transport; estimator code is used unmodified.

#include <boost/make_shared.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "fast_lio.hpp"
#include "fast_lio_debug.hpp"

namespace {
constexpr uint32_t kRequestMagic = 0x324f494c;   // LIO2
constexpr uint32_t kResponseMagic = 0x32534f50;  // POS2

#pragma pack(push, 1)
struct RequestHeader {
  uint32_t magic;
  uint32_t version;
  double scan_start_s;
  uint32_t imu_count;
  uint32_t point_count;
};
struct ImuSample {
  double timestamp_s;
  double gyro[3];
  double acceleration[3];
};
struct LidarPoint {
  float x;
  float y;
  float z;
  uint32_t offset_ns;
};
struct ResponseHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t ready;
  uint32_t point_count;
  double pose[7];  // xyz, quaternion xyzw
};
struct RegisteredPoint {
  float x;
  float y;
  float z;
  float intensity;
};
#pragma pack(pop)

template <typename T>
bool read_exact(T* value, size_t count = 1) {
  return std::fread(value, sizeof(T), count, stdin) == count;
}

template <typename T>
bool write_exact(const T* value, size_t count = 1) {
  return std::fwrite(value, sizeof(T), count, stdout) == count;
}
}  // namespace

int main() {
  // These are the DimOS Mid-360 defaults, tightened to the simulated 16 m
  // warehouse and 10 cm downstream display resolution.
  FastLioParams params;
  params.lidar_type = 1;
  params.scan_line = 4;
  params.scan_rate = 10;
  params.timestamp_unit = 3;
  params.blind = 0.35;
  params.det_range = 20.0;
  params.fov_degree = 360;
  params.extrinsic_est_en = false;
  params.gravity_align = true;
  params.filter_size_surf = 0.10;
  params.filter_size_map = 0.10;
  fastlio_debug = false;
  FastLio estimator(params, 50.0, 5000.0);

  while (true) {
    RequestHeader request{};
    if (!read_exact(&request)) return std::feof(stdin) ? 0 : 2;
    if (request.magic != kRequestMagic || request.version != 1 ||
        request.imu_count > 1000 || request.point_count > 200000) {
      std::fprintf(stderr, "invalid FAST-LIO2 simulation request\n");
      return 3;
    }
    std::vector<ImuSample> imu(request.imu_count);
    std::vector<LidarPoint> points(request.point_count);
    if ((!imu.empty() && !read_exact(imu.data(), imu.size())) ||
        (!points.empty() && !read_exact(points.data(), points.size()))) return 2;

    for (const auto& sample : imu) {
      auto message = boost::make_shared<custom_messages::Imu>();
      message->header.stamp = custom_messages::Time().fromSec(sample.timestamp_s);
      message->orientation.x = message->orientation.y = message->orientation.z = 0.0;
      message->orientation.w = 1.0;
      message->angular_velocity.x = sample.gyro[0];
      message->angular_velocity.y = sample.gyro[1];
      message->angular_velocity.z = sample.gyro[2];
      message->linear_acceleration.x = sample.acceleration[0];
      message->linear_acceleration.y = sample.acceleration[1];
      message->linear_acceleration.z = sample.acceleration[2];
      estimator.feed_imu(message);
    }

    auto scan = boost::make_shared<custom_messages::CustomMsg>();
    scan->header.stamp = custom_messages::Time().fromSec(request.scan_start_s);
    scan->timebase = static_cast<ulli>(request.scan_start_s * 1e9);
    scan->lidar_id = 0;
    scan->point_num = static_cast<uli>(points.size());
    scan->points.reserve(points.size());
    for (const auto& point : points) {
      custom_messages::CustomPoint converted{};
      converted.x = point.x;
      converted.y = point.y;
      converted.z = point.z;
      converted.offset_time = point.offset_ns;
      converted.reflectivity = 100;
      converted.tag = 0;
      converted.line = 0;
      scan->points.push_back(converted);
    }
    estimator.feed_lidar(scan);
    // One call consumes one synchronized lidar/IMU package. A second cheap
    // call handles a package that became ready at the exact scan boundary.
    estimator.process();
    estimator.process();

    ResponseHeader response{};
    response.magic = kResponseMagic;
    response.version = 1;
    const auto pose = estimator.get_pose();
    auto registered = estimator.get_body_cloud_down();
    response.ready = registered && !registered->empty() ? 1U : 0U;
    response.point_count = response.ready ? static_cast<uint32_t>(registered->size()) : 0U;
    for (size_t index = 0; index < 7 && index < pose.size(); ++index) response.pose[index] = pose[index];
    if (!write_exact(&response)) return 2;
    if (response.ready) {
      std::vector<RegisteredPoint> output;
      output.reserve(registered->size());
      for (const auto& point : registered->points) {
        output.push_back({point.x, point.y, point.z, point.intensity});
      }
      if (!write_exact(output.data(), output.size())) return 2;
    }
    std::fflush(stdout);
  }
}
