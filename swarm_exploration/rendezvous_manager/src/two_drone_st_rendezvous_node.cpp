
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <exploration_manager/DroneState.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace rendezvous_manager {

struct DroneIntentState {
  int id = -1;
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d intent_dir = Eigen::Vector3d::UnitX();
  Eigen::Vector3d intent_progress_vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d intent_pred_pos = Eigen::Vector3d::Zero();
  double stamp = 0.0;
  double recv_stamp = 0.0;
  double intent_conf = 0.0;
  double intent_sigma_rad = 0.0;
  double intent_t_int = 0.0;
  double intent_aligned_speed = 0.0;
  double intent_aligned_speed_signed = 0.0;
  double intent_speed_window = 0.0;
  double intent_update_stamp = 0.0;
  double intent_pred_horizon = 0.0;
  bool intent_valid = false;
};

class TwoDroneSTRendezvousNode {
public:
  explicit TwoDroneSTRendezvousNode(ros::NodeHandle& nh) : nh_(nh), pnh_("~") {
    pnh_.param<std::string>("state_topic", state_topic_, std::string("/swarm_expl/drone_state"));
    pnh_.param<std::string>("marker_topic", marker_topic_, std::string("/rendezvous_manager/two_drone_region"));
    pnh_.param<std::string>("frame_id", frame_id_, std::string("world"));
    pnh_.param("drone_id_1", drone_id_1_, 1);
    pnh_.param("drone_id_2", drone_id_2_, 2);
    pnh_.param("self_drone_id", self_drone_id_, drone_id_1_);
    pnh_.param("leader_drone_id", leader_drone_id_, drone_id_1_);
    pnh_.param("compute_only_if_leader", compute_only_if_leader_, true);
    pnh_.param("log_nonleader_standby", log_nonleader_standby_, true);
    pnh_.param("min_time", min_time_, 1.0);
    pnh_.param("max_time", max_time_, 12.0);
    pnh_.param("default_time", default_time_, 3.0);
    pnh_.param("region_radius", region_radius_, 0.0);
    pnh_.param("comm_range", comm_range_, 12.0);
    pnh_.param("comm_margin", comm_margin_, 0.5);
    pnh_.param("min_intent_speed", min_intent_speed_, 0.05);
    pnh_.param("min_intent_conf", min_intent_conf_, 0.10);
    pnh_.param("min_intent_integral_time", min_intent_integral_time_, 1.0);
    pnh_.param("state_timeout", state_timeout_, 5.0);
    pnh_.param("publish_rate", publish_rate_, 5.0);
    pnh_.param("enable_comm_trigger", enable_comm_trigger_, true);
    pnh_.param("force_generate_for_debug", force_generate_for_debug_, false);
    pnh_.param("comm_trigger_ratio", comm_trigger_ratio_, 0.85);
    pnh_.param("comm_trigger_time", comm_trigger_time_, 6.0);
    pnh_.param("region_hold_time", region_hold_time_, 30.0);
    pnh_.param("min_regenerate_interval", min_regenerate_interval_, 5.0);
    pnh_.param("keep_last_region_when_waiting", keep_last_region_when_waiting_, true);
    pnh_.param("vis_z_offset", vis_z_offset_, 0.25);
    pnh_.param("center_scale", center_scale_, 0.35);
    pnh_.param("circle_segments", circle_segments_, 72);
    pnh_.param("show_waiting_marker", show_waiting_marker_, true);
    pnh_.param("debug_log", debug_log_, true);

    if (drone_id_1_ == drone_id_2_) {
      ROS_WARN("[rendezvous_manager] drone_id_1 equals drone_id_2. Set drone_id_2 to drone_id_1 + 1.");
      drone_id_2_ = drone_id_1_ + 1;
    }
    if (self_drone_id_ <= 0) self_drone_id_ = drone_id_1_;
    if (leader_drone_id_ <= 0) leader_drone_id_ = drone_id_1_;
    if (leader_drone_id_ != drone_id_1_ && leader_drone_id_ != drone_id_2_) {
      ROS_WARN("[rendezvous_manager] leader_drone_id is not in [drone_id_1, drone_id_2]. Use drone_id_1 as leader.");
      leader_drone_id_ = drone_id_1_;
    }
    max_time_ = std::max(min_time_ + 0.1, max_time_);
    default_time_ = clamp(default_time_, min_time_, max_time_);

    state_sub_ = nh_.subscribe(state_topic_, 50, &TwoDroneSTRendezvousNode::stateCallback, this);
    // Latch the latest MarkerArray so RViz can display it even if the display is added after startup.
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(marker_topic_, 2, true);
    const double rate = std::max(0.5, publish_rate_);
    timer_ = nh_.createTimer(ros::Duration(1.0 / rate), &TwoDroneSTRendezvousNode::timerCallback, this);

    ROS_INFO_STREAM("[rendezvous_manager] Two-drone ST rendezvous node started. state_topic="
                    << state_topic_ << ", marker_topic=" << marker_topic_
                    << ", drones=[" << drone_id_1_ << "," << drone_id_2_ << "]"
                    << ", self=" << self_drone_id_ << ", leader=" << leader_drone_id_
                    << ", compute_only_if_leader=" << (compute_only_if_leader_ ? "true" : "false"));
  }

private:
  static double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
  }

  static bool normalizeSafe(Eigen::Vector3d& v) {
    const double n = v.norm();
    if (n < 1e-8) return false;
    v /= n;
    return true;
  }

  static Eigen::Vector3d vecFromArray(const std::vector<float>& arr, const Eigen::Vector3d& fallback) {
    if (arr.size() >= 3) return Eigen::Vector3d(arr[0], arr[1], arr[2]);
    return fallback;
  }

  void stateCallback(const exploration_manager::DroneStateConstPtr& msg) {
    if (msg->drone_id <= 0) return;
    DroneIntentState s;
    s.id = msg->drone_id;
    s.stamp = msg->stamp;
    s.recv_stamp = ros::Time::now().toSec();
    s.pos = vecFromArray(msg->pos, Eigen::Vector3d::Zero());
    s.vel = vecFromArray(msg->vel, Eigen::Vector3d::Zero());
    s.intent_valid = msg->intent_valid;
    s.intent_dir = vecFromArray(msg->intent_dir, Eigen::Vector3d::UnitX());
    if (!normalizeSafe(s.intent_dir)) s.intent_dir = Eigen::Vector3d::UnitX();
    s.intent_conf = msg->intent_conf;
    s.intent_sigma_rad = msg->intent_sigma_rad;
    s.intent_t_int = msg->intent_t_int;
    s.intent_aligned_speed = msg->intent_aligned_speed;
    s.intent_aligned_speed_signed = msg->intent_aligned_speed_signed;
    s.intent_speed_window = msg->intent_speed_window;
    s.intent_update_stamp = msg->intent_update_stamp;
    s.intent_pred_horizon = msg->intent_pred_horizon;
    s.intent_progress_vel = vecFromArray(msg->intent_progress_vel, s.intent_dir * std::max(0.0, s.intent_aligned_speed));
    s.intent_pred_pos = vecFromArray(msg->intent_pred_pos, s.pos + s.intent_progress_vel * s.intent_pred_horizon);
    states_[s.id] = s;
  }

  Eigen::Vector3d getProgressVelocity(const DroneIntentState& s) const {
    Eigen::Vector3d v = s.intent_progress_vel;
    if (v.norm() > 1e-5) return v;
    Eigen::Vector3d dir = s.intent_dir;
    if (!normalizeSafe(dir)) return Eigen::Vector3d::Zero();
    return dir * std::max(0.0, s.intent_aligned_speed);
  }

  double getRegionRadius() const {
    if (region_radius_ > 1e-3) return std::max(0.1, region_radius_);
    if (comm_range_ > 1e-3 && comm_range_ < 1e8) {
      return 0.5 * std::max(0.1, comm_range_ - comm_margin_);
    }
    return 3.0;
  }

  visualization_msgs::Marker baseMarker(const std::string& ns, int id, int type) const {
    visualization_msgs::Marker m;
    m.header.frame_id = frame_id_;
    m.header.stamp = ros::Time::now();
    m.ns = ns;
    m.id = id;
    m.type = type;
    m.action = visualization_msgs::Marker::ADD;
    m.pose.orientation.w = 1.0;
    // Persistent markers make debugging easier; they are overwritten by the next publish cycle
    // and cleared explicitly with DELETEALL when the state becomes invalid.
    m.lifetime = ros::Duration(0.0);
    return m;
  }

  static void setColor(visualization_msgs::Marker& m, double r, double g, double b, double a) {
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
  }

  static geometry_msgs::Point toPoint(const Eigen::Vector3d& p) {
    geometry_msgs::Point q;
    q.x = p.x(); q.y = p.y(); q.z = p.z();
    return q;
  }

  visualization_msgs::Marker makeTextMarker(const Eigen::Vector3d& pos, const std::string& text,
                                            double r, double g, double b, double a) const {
    auto m = baseMarker("rv_text", 10, visualization_msgs::Marker::TEXT_VIEW_FACING);
    m.pose.position = toPoint(pos);
    m.scale.z = 0.34;
    setColor(m, r, g, b, a);
    m.text = text;
    return m;
  }

  visualization_msgs::Marker makeDeleteAllMarker() const {
    visualization_msgs::Marker m;
    m.header.frame_id = frame_id_;
    m.header.stamp = ros::Time::now();
    m.action = visualization_msgs::Marker::DELETEALL;
    return m;
  }


  double predictTimeToCommBoundary(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                                   const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) const {
    if (comm_range_ <= 1e-3 || comm_range_ > 1e8) return std::numeric_limits<double>::infinity();
    Eigen::Vector2d r((p1 - p2).x(), (p1 - p2).y());
    Eigen::Vector2d v((v1 - v2).x(), (v1 - v2).y());
    const double R = comm_range_;
    const double c = r.squaredNorm() - R * R;
    if (c >= 0.0) return 0.0;
    const double a = v.squaredNorm();
    if (a < 1e-8) return std::numeric_limits<double>::infinity();
    const double b = 2.0 * r.dot(v);
    const double disc = b * b - 4.0 * a * c;
    if (disc < 0.0) return std::numeric_limits<double>::infinity();
    const double sqrt_disc = std::sqrt(disc);
    const double t1 = (-b - sqrt_disc) / (2.0 * a);
    const double t2 = (-b + sqrt_disc) / (2.0 * a);
    double ans = std::numeric_limits<double>::infinity();
    if (t1 >= 0.0) ans = std::min(ans, t1);
    if (t2 >= 0.0) ans = std::min(ans, t2);
    return ans;
  }

  std::string formatTriggerStatus(const DroneIntentState& s1, const DroneIntentState& s2,
                                  double dist_now, double t_disc, bool triggered) const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "comm_dist=" << dist_now << " / " << comm_range_ << "m";
    ss << "\ntrigger_ratio=" << comm_trigger_ratio_ << "  trigger_time=" << comm_trigger_time_ << "s";
    ss << "\nt_disconnect=";
    if (std::isfinite(t_disc)) ss << t_disc << "s";
    else ss << "inf";
    ss << "  triggered=" << (triggered ? "true" : "false");
    ss << "\nintent_conf=(" << s1.intent_conf << "," << s2.intent_conf << ")";
    ss << "  t_int=(" << s1.intent_t_int << "," << s2.intent_t_int << ")";
    return ss.str();
  }

  void publishWaitingText(const std::string& reason, const std::string& extra = std::string()) {
    if (debug_log_) {
      ROS_WARN_STREAM_THROTTLE(1.0, "[rendezvous_manager] rendezvous region not generated: " << reason
                               << "; received_state_count=" << states_.size()
                               << "; required_drones=[" << drone_id_1_ << "," << drone_id_2_ << "]"
                               << (extra.empty() ? "" : ("; " + extra)));
    }

    visualization_msgs::MarkerArray arr;
    arr.markers.push_back(makeDeleteAllMarker());
    if (!show_waiting_marker_) {
      marker_pub_.publish(arr);
      return;
    }

    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    int cnt = 0;
    auto it1 = states_.find(drone_id_1_);
    auto it2 = states_.find(drone_id_2_);
    if (it1 != states_.end()) { p += it1->second.pos; ++cnt; }
    if (it2 != states_.end()) { p += it2->second.pos; ++cnt; }
    if (cnt > 0) p /= static_cast<double>(cnt);
    p.z() += 2.0;

    std::ostringstream ss;
    ss << "ST-RV[" << drone_id_1_ << "," << drone_id_2_ << "] waiting"
       << "\nleader=" << leader_drone_id_ << "  self=" << self_drone_id_
       << "\nreason: " << reason
       << "\ntopic: " << state_topic_
       << "\nmarker: " << marker_topic_;
    arr.markers.push_back(makeTextMarker(p, ss.str(), 1.0, 0.45, 0.1, 1.0));
    marker_pub_.publish(arr);
  }

  void timerCallback(const ros::TimerEvent&) {
    if (compute_only_if_leader_ && self_drone_id_ != leader_drone_id_) {
      if (debug_log_ && log_nonleader_standby_) {
        ROS_INFO_STREAM_THROTTLE(2.0, "[rendezvous_manager] standby: self_drone_id=" << self_drone_id_
                                 << " is not leader_drone_id=" << leader_drone_id_
                                 << "; this instance will not compute or publish rendezvous regions.");
      }
      return;
    }

    const auto it1 = states_.find(drone_id_1_);
    const auto it2 = states_.find(drone_id_2_);
    if (it1 == states_.end() || it2 == states_.end()) {
      publishWaitingText("missing drone state");
      return;
    }

    const DroneIntentState& s1 = it1->second;
    const DroneIntentState& s2 = it2->second;
    const double now = ros::Time::now().toSec();
    const bool fresh1 = now - s1.recv_stamp <= std::max(0.1, state_timeout_);
    const bool fresh2 = now - s2.recv_stamp <= std::max(0.1, state_timeout_);
    if (!fresh1 || !fresh2) {
      std::ostringstream extra;
      extra << std::fixed << std::setprecision(2)
            << "age=(" << (now - s1.recv_stamp) << "," << (now - s2.recv_stamp)
            << ")s timeout=" << state_timeout_ << "s";
      publishWaitingText("stale drone state", extra.str());
      return;
    }

    Eigen::Vector3d p1 = s1.pos;
    Eigen::Vector3d p2 = s2.pos;
    Eigen::Vector3d v1 = getProgressVelocity(s1);
    Eigen::Vector3d v2 = getProgressVelocity(s2);
    v1.z() = 0.0;
    v2.z() = 0.0;

    const double dist_now = (p1 - p2).head<2>().norm();
    const double t_disconnect = predictTimeToCommBoundary(p1, p2, v1, v2);
    const bool ratio_trigger = (comm_range_ > 1e-3 && dist_now >= comm_trigger_ratio_ * comm_range_);
    const bool time_trigger = std::isfinite(t_disconnect) && t_disconnect <= comm_trigger_time_;
    const bool comm_triggered = force_generate_for_debug_ || !enable_comm_trigger_ || ratio_trigger || time_trigger;

    if (!comm_triggered) {
      publishWaitingText("comm trigger not active", formatTriggerStatus(s1, s2, dist_now, t_disconnect, false));
      return;
    }

    if (!s1.intent_valid || !s2.intent_valid) {
      publishWaitingText("invalid intent prediction", formatTriggerStatus(s1, s2, dist_now, t_disconnect, true));
      return;
    }
    // Do NOT block rendezvous generation only because confidence is low.
    // In a communication-risk / disconnected case, a low-confidence rendezvous region
    // is still more useful than no rendezvous region. Confidence is now used as a
    // quality flag and as the center-weight, not as a hard gate.
    const bool low_conf =
        (s1.intent_conf < min_intent_conf_ || s2.intent_conf < min_intent_conf_);
    if (low_conf && debug_log_) {
      ROS_WARN_STREAM_THROTTLE(
          1.0,
          "[rendezvous_manager] intent confidence low but continue generating region; conf=("
              << s1.intent_conf << "," << s2.intent_conf << "), min=" << min_intent_conf_);
    }

    // The integrated intent time is also treated as a quality flag. Keeping this as
    // a hard gate can prevent any emergency/reconnection region from appearing right
    // after a history reset. The generated region is tagged LOW_HISTORY in RViz.
    const bool low_history =
        (s1.intent_t_int < min_intent_integral_time_ || s2.intent_t_int < min_intent_integral_time_);
    if (low_history && debug_log_) {
      ROS_WARN_STREAM_THROTTLE(
          1.0,
          "[rendezvous_manager] intent history still short but continue generating region; t_int=("
              << s1.intent_t_int << "," << s2.intent_t_int << "), min=" << min_intent_integral_time_);
    }

    if (s1.intent_aligned_speed < min_intent_speed_ || s2.intent_aligned_speed < min_intent_speed_) {
      publishWaitingText("intent aligned speed too low", formatTriggerStatus(s1, s2, dist_now, t_disconnect, true));
      return;
    }

    const Eigen::Vector3d rel_p = p1 - p2;
    const Eigen::Vector3d rel_v = v1 - v2;
    double eta = default_time_;
    const double rel_v2 = rel_v.squaredNorm();
    if (rel_v2 > 1e-6) eta = -rel_p.dot(rel_v) / rel_v2;
    eta = clamp(eta, min_time_, max_time_);

    Eigen::Vector3d pred1 = p1 + v1 * eta;
    Eigen::Vector3d pred2 = p2 + v2 * eta;
    const double vis_z = 0.5 * (p1.z() + p2.z()) + vis_z_offset_;
    pred1.z() = vis_z;
    pred2.z() = vis_z;

    const double min_weight = 0.05;
    const double w1 = std::max(min_weight, s1.intent_conf);
    const double w2 = std::max(min_weight, s2.intent_conf);
    Eigen::Vector3d center = (w1 * pred1 + w2 * pred2) / std::max(1e-6, w1 + w2);
    center.z() = vis_z;

    const double radius = getRegionRadius();
    double pair_dist = (pred1 - pred2).head<2>().norm();
    bool feasible = pair_dist <= 2.0 * radius + 1e-6;
    double t1_to_center = (pred1 - center).head<2>().norm() / std::max(1e-3, s1.intent_aligned_speed);
    double t2_to_center = (pred2 - center).head<2>().norm() / std::max(1e-3, s2.intent_aligned_speed);
    double time_residual = std::fabs(t1_to_center - t2_to_center);

    const double cache_age = now - cached_region_stamp_;
    if (has_cached_region_ && cache_age < region_hold_time_) {
      center = cached_center_;
      pred1 = cached_pred1_;
      pred2 = cached_pred2_;
      eta = cached_eta_;
      pair_dist = cached_pair_dist_;
      time_residual = cached_time_residual_;
      feasible = cached_feasible_;
    } else if (!has_cached_region_ || cache_age >= min_regenerate_interval_) {
      cached_center_ = center;
      cached_pred1_ = pred1;
      cached_pred2_ = pred2;
      cached_eta_ = eta;
      cached_radius_ = radius;
      cached_pair_dist_ = pair_dist;
      cached_time_residual_ = time_residual;
      cached_feasible_ = feasible;
      cached_region_stamp_ = now;
      has_cached_region_ = true;
    }

    visualization_msgs::MarkerArray arr;
    arr.markers.push_back(makeDeleteAllMarker());

    double cr = feasible ? 0.0 : 1.0;
    double cg = feasible ? 0.85 : 0.35;
    double cb = feasible ? 1.0 : 0.05;
    // Orange means the region is generated from low-confidence or short-history intent.
    // Red still means the two predicted states are not time-synchronized within one region.
    if (feasible && (low_conf || low_history)) {
      cr = 1.0;
      cg = 0.65;
      cb = 0.0;
    }

    auto circle = baseMarker("rv_region_circle", 1, visualization_msgs::Marker::LINE_STRIP);
    circle.scale.x = 0.055;
    setColor(circle, cr, cg, cb, 0.85);
    const int n = std::max(12, circle_segments_);
    for (int i = 0; i <= n; ++i) {
      const double a = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n);
      Eigen::Vector3d p = center;
      p.x() += radius * std::cos(a);
      p.y() += radius * std::sin(a);
      circle.points.push_back(toPoint(p));
    }
    arr.markers.push_back(circle);

    auto center_marker = baseMarker("rv_center", 2, visualization_msgs::Marker::SPHERE);
    center_marker.pose.position = toPoint(center);
    const double center_scale = std::max(0.05, center_scale_);
    center_marker.scale.x = center_scale;
    center_marker.scale.y = center_scale;
    center_marker.scale.z = center_scale;
    setColor(center_marker, cr, feasible ? std::max(0.25, cg) : 0.25, feasible ? std::max(0.0, cb) : 0.0, 1.0);
    arr.markers.push_back(center_marker);

    auto pred_marker = baseMarker("rv_pred_points", 3, visualization_msgs::Marker::SPHERE_LIST);
    pred_marker.scale.x = 0.22;
    pred_marker.scale.y = 0.22;
    pred_marker.scale.z = 0.22;
    setColor(pred_marker, 1.0, 1.0, 0.1, 1.0);
    pred_marker.points.push_back(toPoint(pred1));
    pred_marker.points.push_back(toPoint(pred2));
    arr.markers.push_back(pred_marker);

    auto lines = baseMarker("rv_pred_to_center", 4, visualization_msgs::Marker::LINE_LIST);
    lines.scale.x = 0.025;
    setColor(lines, 0.85, 0.85, 0.85, 0.85);
    lines.points.push_back(toPoint(pred1)); lines.points.push_back(toPoint(center));
    lines.points.push_back(toPoint(pred2)); lines.points.push_back(toPoint(center));
    arr.markers.push_back(lines);

    auto pair_line = baseMarker("rv_pred_pair", 5, visualization_msgs::Marker::LINE_LIST);
    pair_line.scale.x = 0.025;
    if (feasible) setColor(pair_line, 0.0, 0.9, 0.4, 0.8);
    else setColor(pair_line, 1.0, 0.2, 0.1, 0.8);
    pair_line.points.push_back(toPoint(pred1)); pair_line.points.push_back(toPoint(pred2));
    arr.markers.push_back(pair_line);

    std::ostringstream ss;
    ss << "ST-RV[" << drone_id_1_ << "," << drone_id_2_ << "] ";
    if (low_conf) ss << "LOW_CONF ";
    if (low_history) ss << "LOW_HISTORY ";
    ss << (feasible ? "OK" : "NOT_SYNC")
       << "  leader=" << leader_drone_id_
       << "\neta=" << std::fixed << std::setprecision(1) << eta << "s"
       << "  d12=" << std::setprecision(2) << pair_dist << "m"
       << "  R=" << std::setprecision(2) << radius << "m"
       << "\ncenter=(" << std::setprecision(2) << center.x() << "," << center.y() << ")"
       << "  vI=(" << s1.intent_aligned_speed << "," << s2.intent_aligned_speed << ")"
       << "\nconf=(" << std::setprecision(2) << s1.intent_conf << "," << s2.intent_conf << ")"
       << "  min_conf=" << min_intent_conf_
       << "\nt_int=(" << std::setprecision(2) << s1.intent_t_int << "," << s2.intent_t_int << ")"
       << "  min_t=" << min_intent_integral_time_
       << "  terr~" << std::setprecision(2) << time_residual << "s";
    Eigen::Vector3d text_pos = center;
    text_pos.z() += 1.0;
    arr.markers.push_back(makeTextMarker(text_pos, ss.str(), cr, cg, cb, 1.0));

    if (debug_log_) {
      ROS_INFO_STREAM_THROTTLE(1.0, "[rendezvous_manager] publish ST region: center=("
                               << center.x() << "," << center.y() << "," << center.z()
                               << "), eta=" << eta << ", radius=" << radius
                               << ", pred_dist=" << pair_dist << ", feasible=" << feasible);
    }
    marker_pub_.publish(arr);
  }

  ros::NodeHandle nh_, pnh_;
  ros::Subscriber state_sub_;
  ros::Publisher marker_pub_;
  ros::Timer timer_;
  std::map<int, DroneIntentState> states_;

  std::string state_topic_;
  std::string marker_topic_;
  std::string frame_id_;
  int drone_id_1_ = 1;
  int drone_id_2_ = 2;
  int self_drone_id_ = 1;
  int leader_drone_id_ = 1;
  bool compute_only_if_leader_ = true;
  bool log_nonleader_standby_ = true;
  double min_time_ = 1.0;
  double max_time_ = 12.0;
  double default_time_ = 3.0;
  double region_radius_ = 0.0;
  double comm_range_ = 12.0;
  double comm_margin_ = 0.5;
  double min_intent_speed_ = 0.05;
  double min_intent_conf_ = 0.10;
  double min_intent_integral_time_ = 1.0;
  double state_timeout_ = 5.0;
  double publish_rate_ = 5.0;
  bool enable_comm_trigger_ = true;
  bool force_generate_for_debug_ = false;
  double comm_trigger_ratio_ = 0.85;
  double comm_trigger_time_ = 6.0;
  double region_hold_time_ = 30.0;
  double min_regenerate_interval_ = 5.0;
  bool keep_last_region_when_waiting_ = true;
  double vis_z_offset_ = 0.25;
  double center_scale_ = 0.35;
  int circle_segments_ = 72;
  bool show_waiting_marker_ = true;
  bool debug_log_ = true;

  bool has_cached_region_ = false;
  double cached_region_stamp_ = 0.0;
  double cached_eta_ = 0.0;
  double cached_radius_ = 0.0;
  double cached_pair_dist_ = 0.0;
  double cached_time_residual_ = 0.0;
  bool cached_feasible_ = false;
  Eigen::Vector3d cached_center_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d cached_pred1_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d cached_pred2_ = Eigen::Vector3d::Zero();
};

}  // namespace rendezvous_manager

int main(int argc, char** argv) {
  ros::init(argc, argv, "two_drone_st_rendezvous_node");
  ros::NodeHandle nh;
  rendezvous_manager::TwoDroneSTRendezvousNode node(nh);
  ros::spin();
  return 0;
}
