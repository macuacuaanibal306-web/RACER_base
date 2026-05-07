#include <ros/ros.h>
#include <exploration_manager/fast_exploration_manager.h>
#include <active_perception/frontier_finder.h>
#include <exploration_manager/HGrid.h>
#include <exploration_manager/GridTour.h>
#include <visualization_msgs/Marker.h>
#include <traj_utils/planning_visualization.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <plan_manage/backward.hpp>
//多无人机探索任务中的可视化与数据中继
namespace backward {
backward::SignalHandling sh;
}

using namespace fast_planner;

class ExplGroundNode {
public:
  ExplGroundNode(ros::NodeHandle& nh) {
    expl_.reset(new FastExplorationManager);//创建新的快速探索管理器对象
    expl_->initialize(nh);
    //0.5s定时器，用于刷新点云
    frontier_timer_ = nh.createTimer(ros::Duration(0.5), &ExplGroundNode::frontierCallback, this);
    frontier_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/expl_ground_node/frontier", 10);
    grid_pub_ = nh.advertise<visualization_msgs::Marker>("/expl_ground_node/grid", 10);

    grid_tour_sub_ =//单机网格访问顺序
        nh.subscribe("/swarm_expl/grid_tour_recv", 10, &ExplGroundNode::gridTourCallback, this);
    hgrid_sub_ = nh.subscribe("/swarm_expl/hgrid_recv", 10, &ExplGroundNode::HGridCallback, this);
    //整个分层地图的网格边
    nh.param("exploration/drone_num", drone_num_, 4);

    grid_tour_stamp_.resize(drone_num_);
    for (auto& st : grid_tour_stamp_) st = 0.0;//时间戳初始化
    hgrid_stamp_ = 0.0;
  }

  ~ExplGroundNode() {
  }

private:
  void frontierCallback(const ros::TimerEvent& e) {//定时器回调
    static int delay = 0;
    if (++delay < 5) return;//前5个周期（即2.5s）内等待，用于避免空地图误检前沿

    // Detect frontier
    auto ft = expl_->frontier_finder_;
    ft->searchFrontiers();//扫描地图并标记前沿
    ft->computeFrontiersToVisit();//聚类+过滤
    // ft->updateFrontierCostMatrix();

    vector<vector<Eigen::Vector3d>> ftrs;
    ft->getFrontiers(ftrs);//前沿只读接口

    // Draw frontier as point cloud 转换PCL点云
    pcl::PointXYZ pt;
    pcl::PointCloud<pcl::PointXYZ> cloud;

    for (auto cluster : ftrs) {//外层簇
      for (auto cell : cluster) {//内层体素中心
        pt.x = cell[0];
        pt.y = cell[1];
        pt.z = cell[2];
        cloud.push_back(pt);
      }
    }
    cloud.width = cloud.points.size();//一维点云
    cloud.height = 1;
    cloud.is_dense = true;
    cloud.header.frame_id = "world";
    sensor_msgs::PointCloud2 cloud_msg;//消息发布
    pcl::toROSMsg(cloud, cloud_msg);
    frontier_pub_.publish(cloud_msg);
  }
  //网格地图回调与可视化
  void HGridCallback(const exploration_manager::HGridConstPtr& msg) {
    if (msg->stamp <= hgrid_stamp_ + 1e-4) return;

    visualization_msgs::Marker mk;//可视化各种形状和对象的消息类型
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = 0;
    mk.ns = "hgrid";//ns=hgrid+id
    mk.type = visualization_msgs::Marker::LINE_LIST;

    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;//单位四元数：不旋转
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;

    mk.color.r = 1.0;
    mk.color.g = 0.0;//紫色
    mk.color.b = 1.0;
    mk.color.a = 0.5;

    mk.scale.x = 0.05;
    mk.scale.y = 0.05;//线宽
    mk.scale.z = 0.05;

    mk.action = visualization_msgs::Marker::DELETE;//把 上一帧同名同 id 的 Marker 从 RViz 场景里删掉
    grid_pub_.publish(mk);

    for (int i = 0; i < msg->points1.size(); ++i) {
      mk.points.push_back(msg->points1[i]);//起点
      mk.points.push_back(msg->points2[i]);//终点
    }

    mk.action = visualization_msgs::Marker::ADD;//把填好坐标的 Marker 重新扔进 RViz，完成一帧刷新
    grid_pub_.publish(mk);
    ros::Duration(0.0005).sleep();

    hgrid_stamp_ = msg->stamp;//记录这一帧已处理，下一帧对比用
  }
  //无人机的网格访问路径
  void gridTourCallback(const exploration_manager::GridTourConstPtr& msg) {
    if (msg->stamp <= grid_tour_stamp_[msg->drone_id - 1] + 1e-4) return;//超时
    if (msg->points.empty()) return;//如果飞机当前没规划出任何路径点

    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = msg->drone_id;//飞机id
    mk.ns = "grid tour";
    mk.type = visualization_msgs::Marker::LINE_LIST;

    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;
    //通过无人机id计算颜色，保证不同颜色的区分
    auto color = PlanningVisualization::getColor((msg->drone_id - 1) / double(drone_num_));

    mk.color.r = color(0);
    mk.color.g = color(1);
    mk.color.b = color(2);
    mk.color.a = color(3);

    mk.scale.x = 0.05;
    mk.scale.y = 0.05;
    mk.scale.z = 0.05;

    mk.action = visualization_msgs::Marker::DELETE;
    grid_pub_.publish(mk);
    //把路径点两两连成线段
    for (int i = 0; i < msg->points.size() - 1; ++i) {
      auto pt1 = msg->points[i];
      auto pt2 = msg->points[i + 1];
      mk.points.push_back(pt1);
      mk.points.push_back(pt2);
    }

    mk.action = visualization_msgs::Marker::ADD;
    grid_pub_.publish(mk);
    ros::Duration(0.0005).sleep();

    grid_tour_stamp_[msg->drone_id - 1] = msg->stamp;
  }

  // Data ------------------------
  shared_ptr<FastExplorationManager> expl_;

  ros::Timer frontier_timer_;
  ros::Publisher frontier_pub_, grid_pub_;
  ros::Subscriber grid_tour_sub_, hgrid_sub_;

  vector<double> grid_tour_stamp_;
  double hgrid_stamp_;//记录上次处理网格的时间
  int drone_num_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "ground_node");
  ros::NodeHandle nh("~");

  ExplGroundNode ground_node(nh);

  ros::Duration(1.0).sleep();
  ros::spin();

  return 0;
}
