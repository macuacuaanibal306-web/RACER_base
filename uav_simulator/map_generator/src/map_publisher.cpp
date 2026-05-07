#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <algorithm>
#include <limits>

using namespace std;
string file_name;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "map_recorder");
  ros::NodeHandle node;
  ros::NodeHandle pnh("~");

  ros::Publisher cloud_pub = node.advertise<sensor_msgs::PointCloud2>("/map_generator/global_cloud", 10, true);
  file_name = argv[1];

  ros::Duration(1.0).sleep();

  pcl::PointCloud<pcl::PointXYZ> cloud;
  int status = pcl::io::loadPCDFile<pcl::PointXYZ>(file_name, cloud);
  if (status == -1)
  {
    cout << "can't read file." << endl;
    return -1;
  }

  bool add_ground_plane = true;
  bool ground_use_map_size = true;
  double ground_resolution = 0.1;
  double ground_z = 0.0;
  double ground_margin_xy = 0.0;
  double map_size_x = 0.0, map_size_y = 0.0;
  double map_min_x = 0.0, map_min_y = 0.0;
  double map_max_x = 0.0, map_max_y = 0.0;

  pnh.param("add_ground_plane", add_ground_plane, true);
  pnh.param("ground_use_map_size", ground_use_map_size, true);
  pnh.param("ground_resolution", ground_resolution, 0.1);
  pnh.param("ground_z", ground_z, 0.0);
  pnh.param("ground_margin_xy", ground_margin_xy, 0.0);
  pnh.param("map_size_x", map_size_x, 0.0);
  pnh.param("map_size_y", map_size_y, 0.0);
  pnh.param("map_min_x", map_min_x, 0.0);
  pnh.param("map_min_y", map_min_y, 0.0);
  pnh.param("map_max_x", map_max_x, 0.0);
  pnh.param("map_max_y", map_max_y, 0.0);

  if (add_ground_plane)
  {
    double min_x = 0.0, max_x = 0.0, min_y = 0.0, max_y = 0.0;
    if (ground_use_map_size && map_size_x > 1e-6 && map_size_y > 1e-6)
    {
      if (map_max_x > map_min_x && map_max_y > map_min_y)
      {
        min_x = map_min_x;
        max_x = map_max_x;
        min_y = map_min_y;
        max_y = map_max_y;
      }
      else
      {
        min_x = -0.5 * map_size_x;
        max_x =  0.5 * map_size_x;
        min_y = -0.5 * map_size_y;
        max_y =  0.5 * map_size_y;
      }
      ROS_WARN("[map_pub] adding ground plane from launch map bounds: x=[%.2f,%.2f], y=[%.2f,%.2f], z=%.2f, res=%.2f",
               min_x, max_x, min_y, max_y, ground_z, ground_resolution);
    }
    else
    {
      min_x = numeric_limits<double>::infinity();
      min_y = numeric_limits<double>::infinity();
      max_x = -numeric_limits<double>::infinity();
      max_y = -numeric_limits<double>::infinity();
      for (const auto& pt : cloud.points)
      {
        min_x = std::min(min_x, (double)pt.x);
        max_x = std::max(max_x, (double)pt.x);
        min_y = std::min(min_y, (double)pt.y);
        max_y = std::max(max_y, (double)pt.y);
      }
      min_x -= ground_margin_xy;
      min_y -= ground_margin_xy;
      max_x += ground_margin_xy;
      max_y += ground_margin_xy;
      ROS_WARN("[map_pub] adding adaptive ground plane from loaded map bounds: x=[%.2f,%.2f], y=[%.2f,%.2f], z=%.2f, res=%.2f",
               min_x, max_x, min_y, max_y, ground_z, ground_resolution);
    }

    for (double x = min_x; x <= max_x + 1e-9; x += ground_resolution)
      for (double y = min_y; y <= max_y + 1e-9; y += ground_resolution)
        cloud.push_back(pcl::PointXYZ(x, y, ground_z));
  }

  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(cloud, msg);
  msg.header.frame_id = "world";

  while (ros::ok())
  {
    ros::Duration(0.2).sleep();
    cloud_pub.publish(msg);
  }

  cout << "finish publish map." << endl;
  return 0;
}
