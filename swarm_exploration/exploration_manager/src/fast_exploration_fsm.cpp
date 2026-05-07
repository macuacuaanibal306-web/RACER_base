
#include <plan_manage/planner_manager.h>
#include <exploration_manager/fast_exploration_manager.h>
#include <traj_utils/planning_visualization.h>

#include <exploration_manager/fast_exploration_fsm.h>
#include <exploration_manager/expl_data.h>
#include <exploration_manager/HGrid.h>
#include <exploration_manager/GridTour.h>

#include <plan_env/edt_environment.h>
#include <plan_env/sdf_map.h>
#include <plan_env/multi_map_manager.h>
#include <active_perception/perception_utils.h>
#include <active_perception/hgrid.h>
// #include <active_perception/uniform_grid.h>
// #include <lkh_tsp_solver/lkh_interface.h>
// #include <lkh_mtsp_solver/lkh3_interface.h>

#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <iomanip>
#include <cmath>

using Eigen::Vector4d;


namespace {
std::string getDiagLogDir() {
  std::string dir;
  if (!ros::param::get("/diag/log_dir", dir) || dir.empty()) {
    const char* home = getenv("HOME");
    dir = std::string(home ? home : "/tmp") + "/.ros/swarm_diag_logs";
  }
  return dir;
}

bool getDiagFileLogEnable() {
  bool enable = true;
  ros::param::param<bool>("/diag/file_log_enable", enable, true);
  return enable;
}

void ensureDirRecursive(const std::string& dir) {
  if (dir.empty()) return;
  std::string cur;
  for (char c : dir) {
    cur.push_back(c);
    if (c == '/') {
      if (!cur.empty() && cur != "/") mkdir(cur.c_str(), 0755);
    }
  }
  if (!cur.empty()) mkdir(cur.c_str(), 0755);
}

void appendDiagLine(int drone_id, const std::string& tag, const std::string& line) {
  if (!getDiagFileLogEnable()) return;
  const std::string dir = getDiagLogDir();
  ensureDirRecursive(dir);
  const std::string ts = std::to_string(ros::Time::now().toSec());
  const std::string full = "[" + ts + "][" + tag + "][drone " + std::to_string(drone_id) + "] " + line + "\n";
  {
    std::ofstream ofs(dir + "/swarm_diag_all.txt", std::ios::app);
    if (ofs.good()) ofs << full;
  }
  {
    std::ofstream ofs(dir + "/drone_" + std::to_string(drone_id) + "_diag.txt", std::ios::app);
    if (ofs.good()) ofs << full;
  }
}


double deg2rad(double deg) { return deg * M_PI / 180.0; }

double clamp01FSM(double v) { return std::max(0.0, std::min(1.0, v)); }

double angleBetweenSafe(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  const double na = a.norm();
  const double nb = b.norm();
  if (na < 1e-8 || nb < 1e-8) return M_PI;
  const double c = std::max(-1.0, std::min(1.0, a.dot(b) / (na * nb)));
  return std::acos(c);
}

bool normalizeSafe(Eigen::Vector3d& v) {
  const double n = v.norm();
  if (n < 1e-8) return false;
  v /= n;
  return true;
}

double yawDeg2D(const Eigen::Vector3d& dir) {
  return std::atan2(dir.y(), dir.x()) * 180.0 / M_PI;
}

std::string headingLabel2D(const Eigen::Vector3d& dir) {
  const double yaw_deg = yawDeg2D(dir);
  const double yaw_norm = std::fmod(yaw_deg + 360.0, 360.0);
  static const char* kLabels[8] = {"E", "NE", "N", "NW", "W", "SW", "S", "SE"};
  const int idx = static_cast<int>(std::floor((yaw_norm + 22.5) / 45.0)) % 8;
  return kLabels[idx];
}

Eigen::Vector4d mixColor(const Eigen::Vector4d& a, const Eigen::Vector4d& b, double t) {
  const double u = clamp01FSM(t);
  return (1.0 - u) * a + u * b;
}

double computeTimeToRectBoundary2D(const Eigen::Vector3d& pos, const Eigen::Vector3d& vel,
    const Eigen::Vector3d& box_min, const Eigen::Vector3d& box_max) {
  const double eps = 1e-6;
  const double kInfTime = 1e9;

  const double vx = vel.x();
  const double vy = vel.y();
  double tx = kInfTime, ty = kInfTime;

  if (vx > eps) tx = (box_max.x() - pos.x()) / vx;
  else if (vx < -eps) tx = (box_min.x() - pos.x()) / vx;

  if (vy > eps) ty = (box_max.y() - pos.y()) / vy;
  else if (vy < -eps) ty = (box_min.y() - pos.y()) / vy;

  if (tx <= 0.0) tx = kInfTime;
  if (ty <= 0.0) ty = kInfTime;
  return std::min(tx, ty);
}

void computeIntentAlignedSpeed(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& hist_pos,
    const std::vector<double>& hist_stamp, const std::vector<double>& hist_dist_w,
    const Eigen::Vector3d& current_vel, const Eigen::Vector3d& intent_dir,
    const double now, const double avg_time, const double min_dt,
    double& aligned_speed, double& signed_speed, double& used_window) {
  const double eps = 1e-6;
  aligned_speed = 0.0;
  signed_speed = 0.0;
  used_window = 0.0;

  Eigen::Vector3d dir = intent_dir;
  if (!normalizeSafe(dir)) {
    return;
  }

  const double T = std::max(1e-3, avg_time);
  const double dt_min = std::max(1e-4, min_dt);
  double wsum = 0.0;
  double signed_sum = 0.0;
  double forward_sum = 0.0;
  double first_stamp = now;
  double last_stamp = now;
  bool have_segment = false;

  for (size_t i = 1; i < hist_stamp.size() && i < hist_pos.size(); ++i) {
    const double dt = hist_stamp[i] - hist_stamp[i - 1];
    if (dt < dt_min) continue;

    const double mid_stamp = 0.5 * (hist_stamp[i] + hist_stamp[i - 1]);
    const double age = now - mid_stamp;
    if (age < -1e-3 || age > T) continue;

    const Eigen::Vector3d seg_vel = (hist_pos[i] - hist_pos[i - 1]) / dt;
    const double proj = seg_vel.dot(dir);
    const double dist_w0 = (i - 1 < hist_dist_w.size()) ? hist_dist_w[i - 1] : 1.0;
    const double dist_w1 = (i < hist_dist_w.size()) ? hist_dist_w[i] : 1.0;
    const double dist_w = 0.5 * (dist_w0 + dist_w1);
    const double w = std::max(1e-3, dt) * std::exp(-age / T) * std::max(0.0, dist_w);
    if (w <= 1e-8) continue;

    signed_sum += w * proj;
    forward_sum += w * std::max(0.0, proj);
    wsum += w;
    if (!have_segment) {
      first_stamp = hist_stamp[i - 1];
      last_stamp = hist_stamp[i];
      have_segment = true;
    } else {
      first_stamp = std::min(first_stamp, hist_stamp[i - 1]);
      last_stamp = std::max(last_stamp, hist_stamp[i]);
    }
  }

  if (wsum > 1e-8) {
    signed_speed = signed_sum / wsum;
    aligned_speed = forward_sum / wsum;
    used_window = std::max(0.0, last_stamp - first_stamp);
    return;
  }

  // Fallback for the first few samples: use instantaneous scalar projection.
  const double inst_proj = current_vel.dot(dir);
  signed_speed = inst_proj;
  aligned_speed = std::max(0.0, inst_proj);
  used_window = 0.0;
}

void truncateIntentHistory(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& hist_pos,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& hist_dir,
    std::vector<double>& hist_stamp, std::vector<double>& hist_dist_w,
    const double now, const double keep_recent_time, const int keep_recent_samples) {
  const int min_keep = std::max(1, keep_recent_samples);
  const double recent_time = std::max(0.0, keep_recent_time);
  while (static_cast<int>(hist_stamp.size()) > min_keep &&
         now - hist_stamp.front() > recent_time) {
    hist_pos.erase(hist_pos.begin());
    hist_dir.erase(hist_dir.begin());
    hist_stamp.erase(hist_stamp.begin());
    hist_dist_w.erase(hist_dist_w.begin());
  }
  while (static_cast<int>(hist_stamp.size()) > min_keep) {
    hist_pos.erase(hist_pos.begin());
    hist_dir.erase(hist_dir.begin());
    hist_stamp.erase(hist_stamp.begin());
    hist_dist_w.erase(hist_dist_w.begin());
  }
}


}  // namespace

namespace fast_planner {
void FastExplorationFSM::init(ros::NodeHandle& nh) {
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /*  Fsm param  */
  nh.param("fsm/thresh_replan1", fp_->replan_thresh1_, -1.0);
  nh.param("fsm/thresh_replan2", fp_->replan_thresh2_, -1.0);
  nh.param("fsm/thresh_replan3", fp_->replan_thresh3_, -1.0);
  nh.param("fsm/replan_time", fp_->replan_time_, -1.0);
  nh.param("fsm/attempt_interval", fp_->attempt_interval_, 0.2);
  nh.param("fsm/pair_opt_interval", fp_->pair_opt_interval_, 1.0);
  nh.param("fsm/enable_pair_opt", fp_->enable_pair_opt_, false);
  nh.param("fsm/repeat_send_num", fp_->repeat_send_num_, 10);
  nh.param("communication/enable_constraint", fp_->enable_comm_constraint_, false);
  nh.param("communication/range", fp_->comm_range_, 1e9);
  nh.param("communication/enable_status_vis", fp_->enable_comm_status_vis_, true);
  nh.param("communication/status_timeout", fp_->comm_status_timeout_, 0.25);
  nh.param("intent_prediction/enable_vis", fp_->enable_intent_prediction_vis_, true);
  nh.param("intent_prediction/show_all_vis", fp_->show_all_intent_prediction_vis_, false);
  nh.param("intent_prediction/hist_max_age", fp_->intent_hist_max_age_, 8.0);
  nh.param("intent_prediction/t_int_init", fp_->intent_t_int_init_, 4.0);
  nh.param("intent_prediction/t_int_min", fp_->intent_t_int_min_, 2.0);
  nh.param("intent_prediction/t_int_max", fp_->intent_t_int_max_, 8.0);
  nh.param("intent_prediction/goal_dist_min", fp_->intent_goal_dist_min_, 1.5);
  nh.param("intent_prediction/goal_dist_ref", fp_->intent_goal_dist_ref_, 4.0);
  nh.param("intent_prediction/var_low_deg", fp_->intent_var_low_deg_, 10.0);
  nh.param("intent_prediction/var_high_deg", fp_->intent_var_high_deg_, 20.0);
  nh.param("intent_prediction/stable_deg", fp_->intent_stable_deg_, 8.0);
  nh.param("intent_prediction/stable_delta_deg", fp_->intent_stable_delta_deg_, 10.0);
  nh.param("intent_prediction/shift_deg", fp_->intent_shift_deg_, 30.0);
  nh.param("intent_prediction/reset_count_thresh", fp_->intent_reset_count_thresh_, 4);
  nh.param("intent_prediction/boundary_hit_time_thresh", fp_->intent_boundary_hit_time_thresh_, 0.5);
  nh.param("intent_prediction/boundary_keep_recent_time", fp_->intent_boundary_keep_recent_time_, 0.4);
  nh.param("intent_prediction/boundary_keep_recent_samples", fp_->intent_boundary_keep_recent_samples_, 3);
  nh.param("intent_prediction/boundary_min_speed", fp_->intent_boundary_min_speed_, 0.15);
  nh.param("intent_prediction/hist_vector_length", fp_->intent_hist_vector_length_, 1.8);
  nh.param("intent_prediction/pred_line_length", fp_->intent_pred_line_length_, 6.0);
  nh.param("intent_prediction/max_hist_vis", fp_->intent_max_hist_vis_, 12);
  nh.param("intent_prediction/speed_avg_time", fp_->intent_speed_avg_time_, 3.0);
  nh.param("intent_prediction/speed_min_dt", fp_->intent_speed_min_dt_, 0.05);
  nh.param("intent_prediction/pred_horizon", fp_->intent_pred_horizon_, 3.0);
  nh.param("intent_prediction/enable_speed_vis", fp_->enable_intent_speed_vis_, true);
  nh.param("intent_prediction/speed_vis_scale", fp_->intent_speed_vis_scale_, 1.5);
  nh.param("intent_prediction/speed_vis_min_len", fp_->intent_speed_vis_min_len_, 0.25);
  nh.param("intent_prediction/speed_vis_max_len", fp_->intent_speed_vis_max_len_, 3.0);

  /* Initialize main modules */
  expl_manager_.reset(new FastExplorationManager);
  expl_manager_->initialize(nh);
  visualization_.reset(new PlanningVisualization(nh));

  planner_manager_ = expl_manager_->planner_manager_;
  state_ = EXPL_STATE::INIT;
  fd_->have_odom_ = false;
  fd_->state_str_ = { "INIT", "WAIT_TRIGGER", "PLAN_TRAJ", "PUB_TRAJ", "EXEC_TRAJ", "FINISH", "IDL"
                                                                                              "E" };
  fd_->static_state_ = true;
  fd_->trigger_ = false;
  fd_->last_stop_reason_ = "INIT";
  fd_->last_stop_detail_ = "FSM initialized, waiting for odometry and trigger";
  fd_->last_replan_trigger_.clear();
  fd_->last_stop_time_ = ros::Time::now();
  fd_->avoid_collision_ = false;
  fd_->go_back_ = false;
  fd_->last_heartbeat_time_ = ros::Time::now();
  for (auto& state : expl_manager_->ed_->swarm_state_) {
    state.intent_t_int_ = fp_->intent_t_int_init_;
    state.intent_pred_horizon_ = fp_->intent_pred_horizon_;
  }

  /* Ros sub, pub and timer */
  exec_timer_ = nh.createTimer(ros::Duration(0.01), &FastExplorationFSM::FSMCallback, this);
  safety_timer_ = nh.createTimer(ros::Duration(0.05), &FastExplorationFSM::safetyCallback, this);
  frontier_timer_ = nh.createTimer(ros::Duration(0.5), &FastExplorationFSM::frontierCallback, this);

  trigger_sub_ =
      nh.subscribe("/move_base_simple/goal", 1, &FastExplorationFSM::triggerCallback, this);
  odom_sub_ = nh.subscribe("/odom_world", 1, &FastExplorationFSM::odometryCallback, this);

  replan_pub_ = nh.advertise<std_msgs::Empty>("/planning/replan", 10);
  new_pub_ = nh.advertise<std_msgs::Empty>("/planning/new", 10);
  bspline_pub_ = nh.advertise<bspline::Bspline>("/planning/bspline", 10);

  // Swarm, timer, pub and sub
  drone_state_timer_ =
      nh.createTimer(ros::Duration(0.04), &FastExplorationFSM::droneStateTimerCallback, this);
  drone_state_pub_ =
      nh.advertise<exploration_manager::DroneState>("/swarm_expl/drone_state_send", 10);
  drone_state_sub_ = nh.subscribe(
      "/swarm_expl/drone_state_recv", 10, &FastExplorationFSM::droneStateMsgCallback, this);

  if (fp_->enable_pair_opt_) {
    opt_timer_ = nh.createTimer(ros::Duration(0.20), &FastExplorationFSM::optTimerCallback, this);
  }
  opt_pub_ = nh.advertise<exploration_manager::PairOpt>("/swarm_expl/pair_opt_send", 10);
  opt_sub_ = nh.subscribe("/swarm_expl/pair_opt_recv", 100, &FastExplorationFSM::optMsgCallback,
      this, ros::TransportHints().tcpNoDelay());

  opt_res_pub_ =
      nh.advertise<exploration_manager::PairOptResponse>("/swarm_expl/pair_opt_res_send", 10);
  opt_res_sub_ = nh.subscribe("/swarm_expl/pair_opt_res_recv", 100,
      &FastExplorationFSM::optResMsgCallback, this, ros::TransportHints().tcpNoDelay());

  swarm_traj_pub_ = nh.advertise<bspline::Bspline>("/planning/swarm_traj_send", 100);
  swarm_traj_sub_ =
      nh.subscribe("/planning/swarm_traj_recv", 100, &FastExplorationFSM::swarmTrajCallback, this);
  swarm_traj_timer_ =
      nh.createTimer(ros::Duration(0.1), &FastExplorationFSM::swarmTrajTimerCallback, this);
  comm_vis_timer_ =
      nh.createTimer(ros::Duration(0.1), &FastExplorationFSM::communicationVisTimerCallback, this);

  hgrid_pub_ = nh.advertise<exploration_manager::HGrid>("/swarm_expl/hgrid_send", 10);
  grid_tour_pub_ = nh.advertise<exploration_manager::GridTour>("/swarm_expl/grid_tour_send", 10);
}

int FastExplorationFSM::getId() const {
  return expl_manager_->ep_->drone_id_;
}

bool FastExplorationFSM::isInCommunicationRange(const Eigen::Vector3d& other_pos) const {
  if (!fp_->enable_comm_constraint_) return true;
  return (fd_->odom_pos_ - other_pos).norm() <= fp_->comm_range_ + 1e-6;
}

Eigen::Vector3d FastExplorationFSM::getCurrentBroadcastPos() const {
  const size_t self_idx = static_cast<size_t>(getId() - 1);
  if (fd_->static_state_ || expl_manager_->ed_->swarm_state_.size() <= self_idx) {
    return fd_->odom_pos_;
  }
  return expl_manager_->ed_->swarm_state_[self_idx].pos_;
}

int FastExplorationFSM::getConnectedDroneCount(std::vector<int>* connected_ids) const {
  if (connected_ids != nullptr) connected_ids->clear();
  if (!fd_->have_odom_) return 0;

  const auto& states = expl_manager_->ed_->swarm_state_;
  const double now = ros::Time::now().toSec();
  const double timeout = std::max(0.05, fp_->comm_status_timeout_);
  int count = 0;

  for (size_t i = 0; i < states.size(); ++i) {
    const int drone_id = static_cast<int>(i) + 1;
    if (drone_id == getId()) continue;

    const auto& state = states[i];
    if (now - state.stamp_ > timeout) continue;
    if (!isInCommunicationRange(state.pos_)) continue;

    ++count;
    if (connected_ids != nullptr) connected_ids->push_back(drone_id);
  }
  return count;
}

void FastExplorationFSM::publishCommunicationStatus() {
  if (!fp_->enable_comm_status_vis_ || !fd_->have_odom_) return;

  std::vector<int> connected_ids;
  const int connected_num = getConnectedDroneCount(&connected_ids);
  const int total_other_num = std::max(0, expl_manager_->ep_->drone_num_ - 1);

  std::ostringstream ss;
  ss << "comm " << connected_num << "/" << total_other_num;
  if (fp_->enable_comm_constraint_) {
    ss << "  r=" << std::fixed << std::setprecision(1) << fp_->comm_range_;
  } else {
    ss << "  free";
  }
  if (!connected_ids.empty()) {
    ss << "\nids:";
    for (size_t i = 0; i < connected_ids.size(); ++i) {
      if (i > 0) ss << ",";
      ss << connected_ids[i];
    }
  }

  Eigen::Vector4d color(1.0, 0.2, 0.2, 1.0);
  if (connected_num >= total_other_num && total_other_num > 0) {
    color = Eigen::Vector4d(0.1, 0.8, 0.1, 1.0);
  } else if (connected_num > 0) {
    color = Eigen::Vector4d(1.0, 0.75, 0.1, 1.0);
  } else if (total_other_num == 0) {
    color = Eigen::Vector4d(0.2, 0.7, 1.0, 1.0);
  }

  Eigen::Vector3d text_pos = getCurrentBroadcastPos();
  text_pos[2] += 0.9;
  visualization_->drawText(text_pos, ss.str(), 0.45, color, "comm_status", 1000 + getId(), 6);
}

void FastExplorationFSM::communicationVisTimerCallback(const ros::TimerEvent& e) {
  publishCommunicationStatus();
  publishIntentVisualization();
}


bool FastExplorationFSM::getDroneIntentGoal(const DroneState& state, Eigen::Vector3d& goal) const {
  if (state.grid_ids_.empty() || expl_manager_ == nullptr || expl_manager_->hgrid_ == nullptr) {
    return false;
  }
  goal = expl_manager_->hgrid_->getCenter(state.grid_ids_.front());
  return true;
}

void FastExplorationFSM::updateDroneIntentPrediction(DroneState& state) {
  const double now = ros::Time::now().toSec();
  Eigen::Vector3d goal;
  if (!getDroneIntentGoal(state, goal)) {
    state.intent_valid_ = false;
    state.intent_goal_ = state.pos_;
    state.intent_aligned_speed_ = 0.0;
    state.intent_aligned_speed_signed_ = 0.0;
    state.intent_speed_window_ = 0.0;
    state.intent_update_stamp_ = now;
    state.intent_pred_horizon_ = fp_->intent_pred_horizon_;
    state.intent_progress_vel_.setZero();
    state.intent_pred_pos_ = state.pos_;
    return;
  }

  Eigen::Vector3d raw = goal - state.pos_;
  const double goal_dist = raw.norm();
  if (goal_dist < 1e-6 || !normalizeSafe(raw)) {
    state.intent_valid_ = false;
    state.intent_goal_ = goal;
    state.intent_aligned_speed_ = 0.0;
    state.intent_aligned_speed_signed_ = 0.0;
    state.intent_speed_window_ = 0.0;
    state.intent_update_stamp_ = now;
    state.intent_pred_horizon_ = fp_->intent_pred_horizon_;
    state.intent_progress_vel_.setZero();
    state.intent_pred_pos_ = state.pos_;
    return;
  }

  const double dist_min = std::max(0.1, fp_->intent_goal_dist_min_);
  const double dist_ref = std::max(dist_min + 1e-3, fp_->intent_goal_dist_ref_);
  const double dist_w = clamp01FSM((goal_dist - dist_min) / (dist_ref - dist_min));

  state.intent_goal_ = goal;
  state.intent_raw_dir_ = raw;
  state.intent_hist_pos_.push_back(state.pos_);
  state.intent_hist_raw_dir_.push_back(raw);
  state.intent_hist_stamp_.push_back(now);
  state.intent_hist_dist_w_.push_back(dist_w);

  const double hist_max_age = std::max(fp_->intent_hist_max_age_, fp_->intent_t_int_max_ + 1.0);
  while (!state.intent_hist_stamp_.empty() && now - state.intent_hist_stamp_.front() > hist_max_age) {
    state.intent_hist_pos_.erase(state.intent_hist_pos_.begin());
    state.intent_hist_raw_dir_.erase(state.intent_hist_raw_dir_.begin());
    state.intent_hist_stamp_.erase(state.intent_hist_stamp_.begin());
    state.intent_hist_dist_w_.erase(state.intent_hist_dist_w_.begin());
  }

  bool boundary_turn_imminent = false;
  const double min_speed = std::max(1e-3, fp_->intent_boundary_min_speed_);
  const double speed_xy = state.vel_.head<2>().norm();
  if (planner_manager_ != nullptr && planner_manager_->edt_environment_ != nullptr &&
      planner_manager_->edt_environment_->sdf_map_ != nullptr && speed_xy >= min_speed) {
    Eigen::Vector3d box_min, box_max;
    planner_manager_->edt_environment_->sdf_map_->getBox(box_min, box_max);
    const double t_hit = computeTimeToRectBoundary2D(state.pos_, state.vel_, box_min, box_max);
    boundary_turn_imminent =
        (t_hit < std::max(0.05, fp_->intent_boundary_hit_time_thresh_));
  }
  if (boundary_turn_imminent && !state.intent_hist_stamp_.empty()) {
    truncateIntentHistory(state.intent_hist_pos_, state.intent_hist_raw_dir_,
        state.intent_hist_stamp_, state.intent_hist_dist_w_, now,
        fp_->intent_boundary_keep_recent_time_, fp_->intent_boundary_keep_recent_samples_);
    state.intent_t_int_ = fp_->intent_t_int_min_;
    state.intent_reset_count_ = 0;
  }

  const double t_int = std::max(fp_->intent_t_int_min_, std::min(fp_->intent_t_int_max_, state.intent_t_int_));
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  double wsum = 0.0;
  for (size_t i = 0; i < state.intent_hist_stamp_.size(); ++i) {
    const double age = now - state.intent_hist_stamp_[i];
    const double w = std::exp(-age / std::max(1e-3, t_int)) * state.intent_hist_dist_w_[i];
    if (w <= 1e-6) continue;
    sum += w * state.intent_hist_raw_dir_[i];
    wsum += w;
  }
  if (wsum <= 1e-6 || !normalizeSafe(sum)) {
    state.intent_valid_ = false;
    state.intent_aligned_speed_ = 0.0;
    state.intent_aligned_speed_signed_ = 0.0;
    state.intent_speed_window_ = 0.0;
    state.intent_update_stamp_ = now;
    state.intent_pred_horizon_ = fp_->intent_pred_horizon_;
    state.intent_progress_vel_.setZero();
    state.intent_pred_pos_ = state.pos_;
    return;
  }

  const Eigen::Vector3d mu = sum;

  double aligned_speed = 0.0;
  double signed_aligned_speed = 0.0;
  double speed_window = 0.0;
  computeIntentAlignedSpeed(state.intent_hist_pos_, state.intent_hist_stamp_, state.intent_hist_dist_w_,
      state.vel_, mu, now, fp_->intent_speed_avg_time_, fp_->intent_speed_min_dt_,
      aligned_speed, signed_aligned_speed, speed_window);

  double sigma2 = 0.0;
  for (size_t i = 0; i < state.intent_hist_stamp_.size(); ++i) {
    const double age = now - state.intent_hist_stamp_[i];
    const double w = std::exp(-age / std::max(1e-3, t_int)) * state.intent_hist_dist_w_[i];
    if (w <= 1e-6) continue;
    const double theta = angleBetweenSafe(state.intent_hist_raw_dir_[i], mu);
    sigma2 += w * theta * theta;
  }
  sigma2 /= std::max(1e-6, wsum);
  const double sigma = std::sqrt(std::max(0.0, sigma2));
  const double var_high_deg = deg2rad(fp_->intent_var_high_deg_);
  if (!state.intent_valid_ || state.intent_hist_stamp_.size() < 3) {
    state.intent_filt_dir_ = mu;
    state.intent_conf_ = clamp01FSM(1.0 - sigma / std::max(1e-3, var_high_deg));
    state.intent_sigma_rad_ = sigma;
    state.intent_aligned_speed_ = aligned_speed;
    state.intent_aligned_speed_signed_ = signed_aligned_speed;
    state.intent_speed_window_ = speed_window;
    state.intent_update_stamp_ = now;
    state.intent_pred_horizon_ = fp_->intent_pred_horizon_;
    state.intent_progress_vel_ = state.intent_filt_dir_ * state.intent_aligned_speed_;
    state.intent_pred_pos_ = state.pos_ + state.intent_progress_vel_ * state.intent_pred_horizon_;
    state.intent_valid_ = true;
    return;
  }
  const double delta = angleBetweenSafe(state.intent_filt_dir_, mu);

  const double stable_deg = deg2rad(fp_->intent_stable_deg_);
  const double stable_delta_deg = deg2rad(fp_->intent_stable_delta_deg_);
  const double shift_deg = deg2rad(fp_->intent_shift_deg_);
  const double var_low_deg = deg2rad(fp_->intent_var_low_deg_);

  if (sigma < stable_deg && delta < stable_delta_deg) {
    state.intent_t_int_ = std::min(fp_->intent_t_int_max_, state.intent_t_int_ + 0.2);
    state.intent_reset_count_ = 0;
  } else if (delta > shift_deg && sigma < var_low_deg) {
    state.intent_reset_count_ += 1;
    if (state.intent_reset_count_ >= std::max(1, fp_->intent_reset_count_thresh_)) {
      state.intent_t_int_ = std::max(fp_->intent_t_int_min_, 0.6 * state.intent_t_int_);
      if (boundary_turn_imminent && !state.intent_hist_stamp_.empty()) {
        truncateIntentHistory(state.intent_hist_pos_, state.intent_hist_raw_dir_,
            state.intent_hist_stamp_, state.intent_hist_dist_w_, now,
            fp_->intent_boundary_keep_recent_time_, fp_->intent_boundary_keep_recent_samples_);
        state.intent_t_int_ = fp_->intent_t_int_min_;
      }
      state.intent_reset_count_ = 0;
    }
  } else {
    state.intent_reset_count_ = std::max(0, state.intent_reset_count_ - 1);
  }
  if (sigma > var_high_deg) {
    state.intent_t_int_ = std::min(fp_->intent_t_int_max_, state.intent_t_int_ + 0.1);
  }

  state.intent_filt_dir_ = mu;
  state.intent_conf_ = clamp01FSM(1.0 - sigma / std::max(1e-3, var_high_deg));
  state.intent_sigma_rad_ = sigma;
  state.intent_aligned_speed_ = aligned_speed;
  state.intent_aligned_speed_signed_ = signed_aligned_speed;
  state.intent_speed_window_ = speed_window;
  state.intent_update_stamp_ = now;
  state.intent_pred_horizon_ = fp_->intent_pred_horizon_;
  state.intent_progress_vel_ = state.intent_filt_dir_ * state.intent_aligned_speed_;
  state.intent_pred_pos_ = state.pos_ + state.intent_progress_vel_ * state.intent_pred_horizon_;
  state.intent_valid_ = true;
}

void FastExplorationFSM::publishIntentVisualization() {
  if (!fp_->enable_intent_prediction_vis_) return;

  const auto& states = expl_manager_->ed_->swarm_state_;
  const double now = ros::Time::now().toSec();
  const bool show_all = fp_->show_all_intent_prediction_vis_;
  const int self_id = getId();

  for (size_t i = 0; i < states.size(); ++i) {
    const int drone_id = static_cast<int>(i) + 1;
    const auto& state = states[i];
    const Eigen::Vector4d base_color = PlanningVisualization::getColor(
        (drone_id - 1) / double(std::max(1, expl_manager_->ep_->drone_num_)), 1.0);
    const Eigen::Vector4d hist_line_color =
        mixColor(base_color, Eigen::Vector4d(1.0, 1.0, 1.0, 1.0), 0.75);
    const Eigen::Vector4d hist_line_color_alpha(
        hist_line_color[0], hist_line_color[1], hist_line_color[2], 0.35);
    const Eigen::Vector4d pred_color = base_color;

    // By default, only show the local drone's intent.
    // If show_all_vis=true, all swarm members' intent markers are visualized.
    if (!show_all && drone_id != self_id) {
      visualization_->drawSpheres({}, 0.03, hist_line_color_alpha, "intent_hist_start", drone_id, 6);
      visualization_->drawLines({}, {}, 0.010, hist_line_color_alpha, "intent_hist", drone_id, 6);
      visualization_->drawArrow(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
          Eigen::Vector3d(0.16, 0.32, 0.42), pred_color, "intent_pred", drone_id, 6);
      visualization_->drawText(Eigen::Vector3d::Zero(), "", 0.35, pred_color, "intent_text", drone_id, 6);
      visualization_->drawArrow(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
          Eigen::Vector3d(0.08, 0.18, 0.24), pred_color, "intent_speed_arrow", drone_id, 6);
      visualization_->drawText(Eigen::Vector3d::Zero(), "", 0.32, pred_color, "intent_speed_text", drone_id, 6);
      continue;
    }

    const bool fresh = (drone_id == self_id) ||
        (now - state.stamp_ <= std::max(0.2, 2.0 * fp_->comm_status_timeout_));

    std::vector<Eigen::Vector3d> hist_start, hist_end;
    Eigen::Vector3d pred_start = Eigen::Vector3d::Zero();
    Eigen::Vector3d pred_end = Eigen::Vector3d::Zero();
    Eigen::Vector3d speed_start = Eigen::Vector3d::Zero();
    Eigen::Vector3d speed_end = Eigen::Vector3d::Zero();
    bool have_pred_arrow = false;
    bool have_speed_vis = false;
    if (fresh && state.intent_valid_) {
      const int hist_n = static_cast<int>(state.intent_hist_stamp_.size());
      const int keep_n = std::max(1, fp_->intent_max_hist_vis_);
      const int begin = std::max(0, hist_n - keep_n);
      for (int k = begin; k < hist_n; ++k) {
        Eigen::Vector3d start_pt = state.intent_hist_pos_[k];
        start_pt[2] += 0.02;
        hist_start.push_back(start_pt);
        Eigen::Vector3d end_pt = start_pt + state.intent_hist_raw_dir_[k] * fp_->intent_hist_vector_length_;
        end_pt[2] = start_pt[2];
        hist_end.push_back(end_pt);
      }

      const double future_len = state.intent_aligned_speed_ * std::max(0.0, state.intent_pred_horizon_);
      const double pred_len = std::max(0.5,
          std::min(fp_->intent_pred_line_length_,
              std::min((state.intent_goal_ - state.pos_).norm(), std::max(0.0, future_len))));
      pred_start = state.pos_;
      pred_start[2] += 0.35;
      pred_end = pred_start + state.intent_filt_dir_ * pred_len;
      pred_end[2] = pred_start[2];
      have_pred_arrow = true;

      const double yaw_deg = yawDeg2D(state.intent_filt_dir_);
      std::ostringstream ss;
      ss << "I" << drone_id << " conf=" << std::fixed << std::setprecision(2) << state.intent_conf_
         << " yaw=" << std::setprecision(1) << yaw_deg << "deg"
         << " dir=(" << std::setprecision(2) << state.intent_filt_dir_.x()
         << "," << state.intent_filt_dir_.y() << ") " << headingLabel2D(state.intent_filt_dir_)
         << "  vI=" << std::setprecision(2) << state.intent_aligned_speed_ << "m/s"
         << "  dt=" << std::setprecision(1) << state.intent_pred_horizon_ << "s"
         << "  T=" << std::setprecision(1) << state.intent_t_int_;
      Eigen::Vector3d text_pos = state.pos_;
      text_pos[2] += 1.25;
      visualization_->drawText(text_pos, ss.str(), 0.35, pred_color, "intent_text", drone_id, 6);

      if (fp_->enable_intent_speed_vis_) {
        const double speed_arrow_len = std::max(fp_->intent_speed_vis_min_len_,
            std::min(fp_->intent_speed_vis_max_len_,
                state.intent_aligned_speed_ * std::max(0.01, fp_->intent_speed_vis_scale_)));
        speed_start = state.pos_;
        speed_start[2] += 0.85;
        speed_end = speed_start + state.intent_filt_dir_ * speed_arrow_len;
        speed_end[2] = speed_start[2];
        have_speed_vis = state.intent_aligned_speed_ > 1e-3;

        std::ostringstream vss;
        vss << "v_pred=" << std::fixed << std::setprecision(2) << state.intent_aligned_speed_ << " m/s"
            << "  win=" << std::setprecision(1) << state.intent_speed_window_ << "s";
        Eigen::Vector3d speed_text_pos = state.pos_;
        speed_text_pos[2] += 1.65;
        visualization_->drawText(speed_text_pos, vss.str(), 0.32,
            Eigen::Vector4d(0.1, 0.9, 1.0, 1.0), "intent_speed_text", drone_id, 6);
      } else {
        visualization_->drawText(Eigen::Vector3d::Zero(), "", 0.32, pred_color, "intent_speed_text", drone_id, 6);
      }
    } else {
      visualization_->drawText(Eigen::Vector3d::Zero(), "", 0.35, base_color, "intent_text", drone_id, 6);
      visualization_->drawText(Eigen::Vector3d::Zero(), "", 0.32, base_color, "intent_speed_text", drone_id, 6);
    }

    visualization_->drawSpheres({}, 0.03, hist_line_color_alpha, "intent_hist_start", drone_id, 6);
    visualization_->drawLines(hist_start, hist_end, 0.010, hist_line_color_alpha, "intent_hist", drone_id, 6);
    visualization_->drawArrow(pred_start, pred_end, Eigen::Vector3d(0.16, 0.32, 0.42), pred_color,
        "intent_pred", drone_id, 6);
    if (!have_pred_arrow) {
      visualization_->drawArrow(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
          Eigen::Vector3d(0.16, 0.32, 0.42), pred_color, "intent_pred", drone_id, 6);
    }

    if (fp_->enable_intent_speed_vis_ && have_speed_vis) {
      visualization_->drawArrow(speed_start, speed_end, Eigen::Vector3d(0.08, 0.18, 0.24),
          Eigen::Vector4d(0.1, 0.9, 1.0, 1.0), "intent_speed_arrow", drone_id, 6);
    } else {
      visualization_->drawArrow(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
          Eigen::Vector3d(0.08, 0.18, 0.24), pred_color, "intent_speed_arrow", drone_id, 6);
    }
  }
}


void FastExplorationFSM::updateStopReason(const string& reason, const string& detail) {
  const bool changed = (fd_->last_stop_reason_ != reason) || (fd_->last_stop_detail_ != detail);
  fd_->last_stop_reason_ = reason;
  fd_->last_stop_detail_ = detail;
  fd_->last_stop_time_ = ros::Time::now();
  if (changed) {
    logStopReason("STOP_DIAG");
  }
}

void FastExplorationFSM::clearStopReason(const string& reason) {
  if (!reason.empty()) {
    ROS_INFO_STREAM("[STOP_DIAG]: Drone " << getId() << " clear stop reason: " << reason);
    appendDiagLine(getId(), "STOP_DIAG", std::string("clear stop reason: ") + reason);
  }
  fd_->last_stop_reason_.clear();
  fd_->last_stop_detail_.clear();
  fd_->last_stop_time_ = ros::Time::now();
}

void FastExplorationFSM::logStopReason(const string& prefix) const {
  const double idle_sec = (ros::Time::now() - fd_->last_stop_time_).toSec();
  std::ostringstream oss;
  oss << "[" << prefix << "]: Drone " << expl_manager_->ep_->drone_id_
      << " state=" << fd_->state_str_[int(state_)]
      << ", static=" << (fd_->static_state_ ? "true" : "false")
      << ", reason=" << (fd_->last_stop_reason_.empty() ? "<none>" : fd_->last_stop_reason_)
      << ", detail=" << (fd_->last_stop_detail_.empty() ? "<none>" : fd_->last_stop_detail_)
      << ", last_replan_trigger="
      << (fd_->last_replan_trigger_.empty() ? "<none>" : fd_->last_replan_trigger_)
      << ", idle_sec=" << idle_sec
      << ", odom=[" << fd_->odom_pos_.transpose() << "]";
  ROS_WARN_STREAM(oss.str());
  appendDiagLine(expl_manager_->ep_->drone_id_, prefix, oss.str());
}

void FastExplorationFSM::logHeartbeat(const string& prefix) const {
  const auto& ed = expl_manager_->ed_;
  const int drone_id = expl_manager_->ep_->drone_id_;
  const int self_idx = std::max(0, drone_id - 1);
  std::vector<int> cur_grid_ids;
  double state_stamp = 0.0;
  if (self_idx < (int)ed->swarm_state_.size()) {
    cur_grid_ids = ed->swarm_state_[self_idx].grid_ids_;
    state_stamp = ed->swarm_state_[self_idx].stamp_;
  }
  auto vecToStr = [](const std::vector<int>& ids) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < ids.size(); ++i) {
      if (i) os << ",";
      os << ids[i];
    }
    os << "]";
    return os.str();
  };
  double traj_remaining = -1.0;
  if (state_ == EXPL_STATE::EXEC_TRAJ || state_ == EXPL_STATE::PUB_TRAJ || state_ == EXPL_STATE::PLAN_TRAJ) {
    auto info = &planner_manager_->local_data_;
    if (info->duration_ > 1e-3) {
      traj_remaining = info->duration_ - (ros::Time::now() - info->start_time_).toSec();
    }
  }
  std::ostringstream oss;
  oss << "state=" << fd_->state_str_[int(state_)]
      << ", static=" << (fd_->static_state_ ? "true" : "false")
      << ", trigger=" << (fd_->trigger_ ? "true" : "false")
      << ", have_odom=" << (fd_->have_odom_ ? "true" : "false")
      << ", avoid_collision=" << (fd_->avoid_collision_ ? "true" : "false")
      << ", go_back=" << (fd_->go_back_ ? "true" : "false")
      << ", stop_reason=" << (fd_->last_stop_reason_.empty() ? "<none>" : fd_->last_stop_reason_)
      << ", last_replan_trigger=" << (fd_->last_replan_trigger_.empty() ? "<none>" : fd_->last_replan_trigger_)
      << ", swarm_stamp=" << state_stamp
      << ", grid_ids=" << vecToStr(cur_grid_ids)
      << ", frontiers=" << ed->frontiers_.size()
      << ", points=" << ed->points_.size()
      << ", next_pos=[" << ed->next_pos_.transpose() << "]"
      << ", next_goal=[" << ed->next_goal_.transpose() << "]"
      << ", traj_remaining=" << traj_remaining
      << ", odom=[" << fd_->odom_pos_.transpose() << "]"
      << ", vel=[" << fd_->odom_vel_.transpose() << "]";
  ROS_INFO_STREAM("[" << prefix << "][drone " << drone_id << "] " << oss.str());
  appendDiagLine(drone_id, prefix, oss.str());
}

void FastExplorationFSM::FSMCallback(const ros::TimerEvent& e) {
  ROS_INFO_STREAM_THROTTLE(
      1.0, "[FSM]: Drone " << getId() << " state: " << fd_->state_str_[int(state_)]);
  if ((ros::Time::now() - fd_->last_heartbeat_time_).toSec() > 1.0) {
    fd_->last_heartbeat_time_ = ros::Time::now();
    logHeartbeat("FSM_HEARTBEAT");
  }

  switch (state_) {
    case INIT: {
      // Wait for odometry ready
      if (!fd_->have_odom_) {
        updateStopReason("WAIT_ODOM", "FSM INIT waiting for odometry message");
        ROS_WARN_THROTTLE(1.0, "no odom");
        return;
      }
      if ((ros::Time::now() - fd_->fsm_init_time_).toSec() < 2.0) {
        updateStopReason("WAIT_INIT_DELAY", "FSM INIT waiting initial stabilization delay");
        ROS_WARN_THROTTLE(1.0, "wait for init");
        return;
      }
      // Go to wait trigger when odom is ok
      updateStopReason("WAIT_TRIGGER", "Odometry ready, waiting user trigger to start exploration");
      transitState(WAIT_TRIGGER, "FSM");
      break;
    }

    case WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      if (fd_->last_stop_reason_ != "WAIT_TRIGGER") {
        updateStopReason("WAIT_TRIGGER", "No trigger received yet");
      }
      ROS_WARN_STREAM_THROTTLE(1.0, "wait for trigger. reason=" << fd_->last_stop_reason_);
      break;
    }

    case FINISH: {
      ROS_INFO_THROTTLE(1.0, "finish exploration.");
      ROS_WARN_STREAM_THROTTLE(2.0, "[STOP_DIAG]: Drone " << getId() << " in FINISH state, reason="
                               << (fd_->last_stop_reason_.empty() ? "exploration finished" : fd_->last_stop_reason_));
      break;
    }

    case IDLE: {
      const ros::Time now = ros::Time::now();
      const double check_interval = (now - fd_->last_check_frontier_time_).toSec();
      const double retry_interval = 1.5;
      ROS_WARN_STREAM_THROTTLE(2.0, "[STOP_DIAG]: Drone " << getId() << " IDLE, reason="
                               << (fd_->last_stop_reason_.empty() ? "<none>" : fd_->last_stop_reason_)
                               << ", detail=" << (fd_->last_stop_detail_.empty() ? "<none>" : fd_->last_stop_detail_)
                               << ", idle_sec=" << check_interval);

      if (check_interval > retry_interval) {
        fd_->last_check_frontier_time_ = now;
        const int frontier_num = expl_manager_->updateFrontierStruct(fd_->odom_pos_);
        if (frontier_num != 0) {
          clearStopReason("idle recovered: frontier available again");
          fd_->go_back_ = false;
          fd_->static_state_ = true;
          ROS_WARN_STREAM("[STOP_DIAG]: Drone " << getId() << " leaves IDLE because frontier_num=" << frontier_num);
          transitState(PLAN_TRAJ, "FSM");
          break;
        }
        updateStopReason("NO_FRONTIER_AFTER_REPLAN",
            std::string("IDLE retry found no coverable frontier after ") + std::to_string(retry_interval) + "s interval");
      }

      if (check_interval > 100.0) {
        updateStopReason("GO_BACK_AFTER_LONG_IDLE", "IDLE timeout exceeded 100s, command return to start position");
        ROS_WARN("Go back to (0,0,1)");

        expl_manager_->ed_->next_pos_ = fd_->start_pos_;
        expl_manager_->ed_->next_yaw_ = 0.0;

        fd_->go_back_ = true;
        transitState(PLAN_TRAJ, "FSM");
      }
      break;
    }

    case PLAN_TRAJ: {
      if (fd_->static_state_) {
        // Plan from static state (hover)
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_vel_ = fd_->odom_vel_;
        fd_->start_acc_.setZero();
        fd_->start_yaw_ << fd_->odom_yaw_, 0, 0;
      } else {
        // Replan from non-static state, starting from 'replan_time' seconds later
        LocalTrajData* info = &planner_manager_->local_data_;
        double t_r = (ros::Time::now() - info->start_time_).toSec() + fp_->replan_time_;
        fd_->start_pt_ = info->position_traj_.evaluateDeBoorT(t_r);
        fd_->start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_r);
        fd_->start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_r);
        fd_->start_yaw_(0) = info->yaw_traj_.evaluateDeBoorT(t_r)[0];
        fd_->start_yaw_(1) = info->yawdot_traj_.evaluateDeBoorT(t_r)[0];
        fd_->start_yaw_(2) = info->yawdotdot_traj_.evaluateDeBoorT(t_r)[0];
      }
      // Inform traj_server the replanning
      replan_pub_.publish(std_msgs::Empty());
      const bool due_to_avoid_collision = fd_->avoid_collision_;
      const bool due_to_go_back = fd_->go_back_;
      int res = callExplorationPlanner();
      if (res == SUCCEED) {
        clearStopReason("planner succeeded");
        transitState(PUB_TRAJ, "FSM");
      } else if (res == FAIL) {  // Keep trying to replan
        fd_->static_state_ = true;
        std::string reason = "PLAN_FAIL";
        std::string detail;
        if (due_to_go_back) {
          detail = "Trajectory planning failed while returning to start position";
        } else if (due_to_avoid_collision) {
          detail = "Trajectory replanning failed after collision-avoidance trigger";
        } else {
          detail = "Exploration planner failed to generate a valid trajectory";
        }
        updateStopReason(reason, detail);
        ROS_WARN("Plan fail");
      } else if (res == NO_GRID) {
        fd_->static_state_ = true;
        fd_->last_check_frontier_time_ = ros::Time::now();
        updateStopReason("NO_GRID", "Planner returned NO_GRID, no allocatable grid for this drone");
        ROS_WARN("No grid");
        transitState(IDLE, "FSM");
        visualize(1);
        // clearVisMarker();
      }
      break;
    }

    case PUB_TRAJ: {
      double dt = (ros::Time::now() - fd_->newest_traj_.start_time).toSec();
      if (dt > 0) {
        bspline_pub_.publish(fd_->newest_traj_);
        fd_->static_state_ = false;

        // fd_->newest_traj_.drone_id = planner_manager_->swarm_traj_data_.drone_id_;
        fd_->newest_traj_.drone_id = expl_manager_->ep_->drone_id_;
        {
          Eigen::Vector3d sender_pos = getCurrentBroadcastPos();
          fd_->newest_traj_.sender_pos = { float(sender_pos[0]), float(sender_pos[1]), float(sender_pos[2]) };
        }
        swarm_traj_pub_.publish(fd_->newest_traj_);

        thread vis_thread(&FastExplorationFSM::visualize, this, 2);
        vis_thread.detach();
        transitState(EXEC_TRAJ, "FSM");
      }
      break;
    }

    case EXEC_TRAJ: {
      auto tn = ros::Time::now();
      // Check whether replan is needed
      LocalTrajData* info = &planner_manager_->local_data_;
      double t_cur = (tn - info->start_time_).toSec();

      if (!fd_->go_back_) {
        bool need_replan = false;
        fd_->last_replan_trigger_.clear();
        if (t_cur > fp_->replan_thresh2_ && expl_manager_->frontier_finder_->isFrontierCovered()) {
          ROS_WARN("Replan: cluster covered=====================================");
          need_replan = true;
          fd_->last_replan_trigger_ = "cluster covered";
        } else if (info->duration_ - t_cur < fp_->replan_thresh1_) {
          // Replan if traj is almost fully executed
          ROS_WARN("Replan: traj fully executed=================================");
          need_replan = true;
          fd_->last_replan_trigger_ = "trajectory almost finished";
        } else if (t_cur > fp_->replan_thresh3_) {
          // Replan after some time
          ROS_WARN("Replan: periodic call=======================================");
          need_replan = true;
          fd_->last_replan_trigger_ = "periodic replanning threshold";
        }

        if (need_replan) {
          if (expl_manager_->updateFrontierStruct(fd_->odom_pos_) != 0) {
            // Update frontier and plan new motion
            thread vis_thread(&FastExplorationFSM::visualize, this, 1);
            vis_thread.detach();
            transitState(PLAN_TRAJ, "FSM");
          } else {
            const double remaining_time = std::max(0.0, info->duration_ - t_cur);
            const bool soft_replan_trigger =
                (fd_->last_replan_trigger_ == "cluster covered") ||
                (fd_->last_replan_trigger_ == "periodic replanning threshold");
            const bool can_keep_current_traj = soft_replan_trigger && remaining_time > std::max(1.0, fp_->replan_thresh1_);

            if (can_keep_current_traj) {
              updateStopReason("NO_FRONTIER_DEFER_REPLAN",
                  std::string("No coverable frontier after soft replan trigger, keep executing current trajectory. trigger=") +
                  fd_->last_replan_trigger_ + ", remaining_time=" + std::to_string(remaining_time));
              ROS_WARN_STREAM("Keep current trajectory since no frontier is detected after soft replan trigger. remaining_time="
                              << remaining_time << ", trigger=" << fd_->last_replan_trigger_);
            } else {
              // No frontier detected, enter IDLE but allow periodic retry in IDLE state.
              fd_->last_check_frontier_time_ = ros::Time::now();
              updateStopReason("NO_FRONTIER_AFTER_REPLAN",
                  std::string("No coverable frontier after replan trigger: ") +
                  (fd_->last_replan_trigger_.empty() ? std::string("unknown") : fd_->last_replan_trigger_));
              transitState(IDLE, "FSM");
              ROS_WARN("Idle since no frontier is detected");
              fd_->static_state_ = true;
              replan_pub_.publish(std_msgs::Empty());
              // clearVisMarker();
              visualize(1);
            }
          }
        }
      } else {
        // Check if reach goal
        auto pos = info->position_traj_.evaluateDeBoorT(t_cur);
        if ((pos - expl_manager_->ed_->next_pos_).norm() < 1.0) {
          replan_pub_.publish(std_msgs::Empty());
          updateStopReason("GO_BACK_TARGET_REACHED", "Returned to start/goal position while go_back_ is active");
          clearVisMarker();
          transitState(FINISH, "FSM");
          return;
        }
        if (t_cur > fp_->replan_thresh3_ || info->duration_ - t_cur < fp_->replan_thresh1_) {
          // Replan for going back
          replan_pub_.publish(std_msgs::Empty());
          transitState(PLAN_TRAJ, "FSM");
          thread vis_thread(&FastExplorationFSM::visualize, this, 1);
          vis_thread.detach();
        }
      }

      break;
    }
  }
}

int FastExplorationFSM::callExplorationPlanner() {
  ros::Time time_r = ros::Time::now() + ros::Duration(fp_->replan_time_);

  int res;
  if (fd_->avoid_collision_ || fd_->go_back_) {  // Only replan trajectory
    res = expl_manager_->planTrajToView(fd_->start_pt_, fd_->start_vel_, fd_->start_acc_,
        fd_->start_yaw_, expl_manager_->ed_->next_pos_, expl_manager_->ed_->next_yaw_);
    fd_->avoid_collision_ = false;
  } else {  // Do full planning normally
    res = expl_manager_->planExploreMotion(
        fd_->start_pt_, fd_->start_vel_, fd_->start_acc_, fd_->start_yaw_);
  }

  if (res == SUCCEED) {
    auto info = &planner_manager_->local_data_;
    info->start_time_ = (ros::Time::now() - time_r).toSec() > 0 ? ros::Time::now() : time_r;

    bspline::Bspline bspline;
    bspline.order = planner_manager_->pp_.bspline_degree_;
    bspline.start_time = info->start_time_;
    bspline.traj_id = info->traj_id_;
    Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
    for (int i = 0; i < pos_pts.rows(); ++i) {
      geometry_msgs::Point pt;
      pt.x = pos_pts(i, 0);
      pt.y = pos_pts(i, 1);
      pt.z = pos_pts(i, 2);
      bspline.pos_pts.push_back(pt);
    }
    Eigen::VectorXd knots = info->position_traj_.getKnot();
    for (int i = 0; i < knots.rows(); ++i) {
      bspline.knots.push_back(knots(i));
    }
    Eigen::MatrixXd yaw_pts = info->yaw_traj_.getControlPoint();
    for (int i = 0; i < yaw_pts.rows(); ++i) {
      double yaw = yaw_pts(i, 0);
      bspline.yaw_pts.push_back(yaw);
    }
    bspline.yaw_dt = info->yaw_traj_.getKnotSpan();
    Eigen::Vector3d sender_pos = getCurrentBroadcastPos();
    bspline.sender_pos = { float(sender_pos[0]), float(sender_pos[1]), float(sender_pos[2]) };
    fd_->newest_traj_ = bspline;
  }
  return res;
}

void FastExplorationFSM::visualize(int content) {
  // content 1: frontier; 2 paths & trajs
  auto info = &planner_manager_->local_data_;
  auto plan_data = &planner_manager_->plan_data_;
  auto ed_ptr = expl_manager_->ed_;

  auto getColorVal = [&](const int& id, const int& num, const int& drone_id) {
    double a = (drone_id - 1) / double(num + 1);
    double b = 1 / double(num + 1);
    return a + b * double(id) / ed_ptr->frontiers_.size();
  };

  if (content == 1) {
    // Draw frontier
    static int last_ftr_num = 0;
    static int last_dftr_num = 0;
    for (int i = 0; i < ed_ptr->frontiers_.size(); ++i) {
      visualization_->drawCubes(ed_ptr->frontiers_[i], 0.1,
          visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 0.4), "frontier", i, 4);

      // getColorVal(i, expl_manager_->ep_->drone_num_, expl_manager_->ep_->drone_id_)
      // double(i) / ed_ptr->frontiers_.size()

      // visualization_->drawBox(ed_ptr->frontier_boxes_[i].first,
      // ed_ptr->frontier_boxes_[i].second,
      //     Vector4d(0.5, 0, 1, 0.3), "frontier_boxes", i, 4);
    }
    for (int i = ed_ptr->frontiers_.size(); i < last_ftr_num; ++i) {
      visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
      // visualization_->drawBox(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector4d(1, 0, 0, 0.3),
      // "frontier_boxes", i, 4);
    }
    last_ftr_num = ed_ptr->frontiers_.size();

    // for (int i = 0; i < ed_ptr->dead_frontiers_.size(); ++i)
    //   visualization_->drawCubes(
    //       ed_ptr->dead_frontiers_[i], 0.1, Vector4d(0, 0, 0, 0.5), "dead_frontier", i, 4);
    // for (int i = ed_ptr->dead_frontiers_.size(); i < last_dftr_num; ++i)
    //   visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 0.5), "dead_frontier", i, 4);
    // last_dftr_num = ed_ptr->dead_frontiers_.size();

    // // Draw updated box
    // Vector3d bmin, bmax;
    // planner_manager_->edt_environment_->sdf_map_->getUpdatedBox(bmin, bmax, false);
    // visualization_->drawBox(
    //     (bmin + bmax) / 2.0, bmax - bmin, Vector4d(0, 1, 0, 0.3), "updated_box", 0, 4);

    // vector<Eigen::Vector3d> bmins, bmaxs;
    // planner_manager_->edt_environment_->sdf_map_->mm_->getChunkBoxes(bmins, bmaxs, false);
    // for (int i = 0; i < bmins.size(); ++i) {
    //   visualization_->drawBox((bmins[i] + bmaxs[i]) / 2.0, bmaxs[i] - bmins[i],
    //       Vector4d(0, 1, 1, 0.3), "updated_box", i + 1, 4);
    // }

  } else if (content == 2) {

    // Hierarchical grid and global tour --------------------------------
    // vector<Eigen::Vector3d> pts1, pts2;
    // expl_manager_->uniform_grid_->getPath(pts1, pts2);
    // visualization_->drawLines(pts1, pts2, 0.05, Eigen::Vector4d(1, 0.3, 0, 1), "partition", 0,
    // 6);

    if (expl_manager_->ep_->drone_id_ == 1) {
      vector<Eigen::Vector3d> pts1, pts2;
      expl_manager_->hgrid_->getGridMarker(pts1, pts2);
      visualization_->drawLines(pts1, pts2, 0.05, Eigen::Vector4d(1, 0, 1, 0.5), "partition", 1, 6);

      vector<Eigen::Vector3d> pts;
      vector<string> texts;
      expl_manager_->hgrid_->getGridMarker2(pts, texts);
      static int last_text_num = 0;
      for (int i = 0; i < pts.size(); ++i) {
        visualization_->drawText(pts[i], texts[i], 1, Eigen::Vector4d(0, 0, 0, 1), "text", i, 6);
      }
      for (int i = pts.size(); i < last_text_num; ++i) {
        visualization_->drawText(
            Eigen::Vector3d(0, 0, 0), string(""), 1, Eigen::Vector4d(0, 0, 0, 1), "text", i, 6);
      }
      last_text_num = pts.size();

      // // Pub hgrid to ground node
      // exploration_manager::HGrid hgrid;
      // hgrid.stamp = ros::Time::now().toSec();
      // for (int i = 0; i < pts1.size(); ++i) {
      //   geometry_msgs::Point pt1, pt2;
      //   pt1.x = pts1[i][0];
      //   pt1.y = pts1[i][1];
      //   pt1.z = pts1[i][2];
      //   hgrid.points1.push_back(pt1);
      //   pt2.x = pts2[i][0];
      //   pt2.y = pts2[i][1];
      //   pt2.z = pts2[i][2];
      //   hgrid.points2.push_back(pt2);
      // }
      // hgrid_pub_.publish(hgrid);
    }

    auto grid_tour = expl_manager_->ed_->grid_tour_;
    // auto grid_tour = expl_manager_->ed_->grid_tour2_;
    // for (auto& pt : grid_tour) pt = pt + trans;

    visualization_->drawLines(grid_tour, 0.05,
        PlanningVisualization::getColor(
            (expl_manager_->ep_->drone_id_ - 1) / double(expl_manager_->ep_->drone_num_)),
        "grid_tour", 0, 6);

    // Publish grid tour to ground node
    exploration_manager::GridTour tour;
    for (int i = 0; i < grid_tour.size(); ++i) {
      geometry_msgs::Point point;
      point.x = grid_tour[i][0];
      point.y = grid_tour[i][1];
      point.z = grid_tour[i][2];
      tour.points.push_back(point);
    }
    tour.drone_id = expl_manager_->ep_->drone_id_;
    tour.stamp = ros::Time::now().toSec();
    grid_tour_pub_.publish(tour);

    // visualization_->drawSpheres(
    //     expl_manager_->ed_->grid_tour_, 0.3, Eigen::Vector4d(0, 1, 0, 1), "grid_tour", 1, 6);
    // visualization_->drawLines(
    //     expl_manager_->ed_->grid_tour2_, 0.05, Eigen::Vector4d(0, 1, 0, 0.5), "grid_tour", 2, 6);

    // Top viewpoints and frontier tour-------------------------------------

    // visualization_->drawSpheres(ed_ptr->points_, 0.2, Vector4d(0, 0.5, 0, 1), "points", 0, 6);
    // visualization_->drawLines(
    //     ed_ptr->points_, ed_ptr->views_, 0.05, Vector4d(0, 1, 0.5, 1), "view", 0, 6);
    // visualization_->drawLines(
    //     ed_ptr->points_, ed_ptr->averages_, 0.03, Vector4d(1, 0, 0, 1), "point-average", 0, 6);

    // auto frontier = ed_ptr->frontier_tour_;
    // for (auto& pt : frontier) pt = pt + trans;
    // visualization_->drawLines(frontier, 0.07,
    //     PlanningVisualization::getColor(
    //         (expl_manager_->ep_->drone_id_ - 1) / double(expl_manager_->ep_->drone_num_), 0.6),
    //     "frontier_tour", 0, 6);

    // for (int i = 0; i < ed_ptr->other_tours_.size(); ++i) {
    //   visualization_->drawLines(
    //       ed_ptr->other_tours_[i], 0.07, Eigen::Vector4d(0, 0, 1, 1), "other_tours", i, 6);
    // }

    // Locally refined viewpoints and refined tour-------------------------------

    // visualization_->drawSpheres(
    //     ed_ptr->refined_points_, 0.2, Vector4d(0, 0, 1, 1), "refined_pts", 0, 6);
    // visualization_->drawLines(
    //     ed_ptr->refined_points_, ed_ptr->refined_views_, 0.05, Vector4d(0.5, 0, 1, 1),
    //     "refined_view", 0, 6);
    // visualization_->drawLines(
    //     ed_ptr->refined_tour_, 0.07,
    //     PlanningVisualization::getColor(
    //         (expl_manager_->ep_->drone_id_ - 1) / double(expl_manager_->ep_->drone_num_), 0.6),
    //     "refined_tour", 0, 6);

    // visualization_->drawLines(ed_ptr->refined_views1_, ed_ptr->refined_views2_, 0.04, Vector4d(0,
    // 0, 0, 1),
    //                           "refined_view", 0, 6);
    // visualization_->drawLines(ed_ptr->refined_points_, ed_ptr->unrefined_points_, 0.05,
    // Vector4d(1, 1, 0, 1),
    //                           "refine_pair", 0, 6);
    // for (int i = 0; i < ed_ptr->n_points_.size(); ++i)
    //   visualization_->drawSpheres(ed_ptr->n_points_[i], 0.1,
    //                               visualization_->getColor(double(ed_ptr->refined_ids_[i]) /
    //                               ed_ptr->frontiers_.size()),
    //                               "n_points", i, 6);
    // for (int i = ed_ptr->n_points_.size(); i < 15; ++i)
    //   visualization_->drawSpheres({}, 0.1, Vector4d(0, 0, 0, 1), "n_points", i, 6);

    // Trajectory-------------------------------------------

    // visualization_->drawSpheres(
    //     { ed_ptr->next_goal_ /* + trans */ }, 0.3, Vector4d(0, 0, 1, 1), "next_goal", 0, 6);

    // vector<Eigen::Vector3d> next_yaw_vis;
    // next_yaw_vis.push_back(ed_ptr->next_goal_ /* + trans */);
    // next_yaw_vis.push_back(
    //     ed_ptr->next_goal_ /* + trans */ +
    //     2.0 * Eigen::Vector3d(cos(ed_ptr->next_yaw_), sin(ed_ptr->next_yaw_), 0));
    // visualization_->drawLines(next_yaw_vis, 0.1, Eigen::Vector4d(0, 0, 1, 1), "next_goal", 1, 6);
    // visualization_->drawSpheres(
    //     { ed_ptr->next_pos_ /* + trans */ }, 0.3, Vector4d(0, 1, 0, 1), "next_pos", 0, 6);

    // Eigen::MatrixXd ctrl_pt = info->position_traj_.getControlPoint();
    // for (int i = 0; i < ctrl_pt.rows(); ++i) {
    //   for (int j = 0; j < 3; ++j) ctrl_pt(i, j) = ctrl_pt(i, j) + trans[j];
    // }
    // NonUniformBspline position_traj(ctrl_pt, 3, info->position_traj_.getKnotSpan());

    visualization_->drawBspline(info->position_traj_, 0.1,
        PlanningVisualization::getColor(
            (expl_manager_->ep_->drone_id_ - 1) / double(expl_manager_->ep_->drone_num_)),
        false, 0.15, Vector4d(1, 1, 0, 1));

    // visualization_->drawLines(
    //     expl_manager_->ed_->path_next_goal_, 0.1, Eigen::Vector4d(0, 1, 0, 1), "astar", 0, 6);
    // visualization_->drawSpheres(
    //     expl_manager_->ed_->kino_path_, 0.1, Eigen::Vector4d(0, 0, 1, 1), "kino", 0, 6);
    // visualization_->drawSpheres(plan_data->kino_path_, 0.1, Vector4d(1, 0, 1, 1), "kino_path", 0,
    // 0); visualization_->drawLines(ed_ptr->path_next_goal_, 0.05, Vector4d(0, 1, 1, 1),
    // "next_goal", 1, 6);

    // // Draw trajs of other drones
    // vector<NonUniformBspline> trajs;
    // planner_manager_->swarm_traj_data_.getValidTrajs(trajs);
    // for (int k = 0; k < trajs.size(); ++k) {
    //   visualization_->drawBspline(trajs[k], 0.1, Eigen::Vector4d(1, 1, 0, 1), false, 0.15,
    //       Eigen::Vector4d(0, 0, 1, 1), k + 1);
    // }
  }
}

void FastExplorationFSM::clearVisMarker() {
  const vector<Vector3d> empty_pts;
  for (int i = 0; i < 10; ++i) {
    visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
    // visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "dead_frontier", i, 4);
    // visualization_->drawBox(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector4d(1, 0, 0, 0.3),
    // "frontier_boxes", i, 4);
  }
  // visualization_->drawSpheres({}, 0.2, Vector4d(0, 0.5, 0, 1), "points", 0, 6);
  visualization_->drawLines({}, 0.07, Vector4d(0, 0.5, 0, 1), "frontier_tour", 0, 6);
  visualization_->drawLines({}, 0.07, Vector4d(0, 0.5, 0, 1), "grid_tour", 0, 6);
  for (int i = 1; i <= expl_manager_->ep_->drone_num_; ++i) {
    visualization_->drawSpheres({}, 0.03, Vector4d(0, 0.5, 0, 1), "intent_hist_start", i, 6);
    visualization_->drawLines(empty_pts, empty_pts, 0.010, Vector4d(0, 0.5, 0, 1), "intent_hist", i, 6);
    visualization_->drawArrow(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Vector3d(0.16, 0.32, 0.42),
        Vector4d(0, 0.5, 0, 1), "intent_pred", i, 6);
    visualization_->drawText(Eigen::Vector3d::Zero(), "", 0.35, Vector4d(0, 0.5, 0, 1), "intent_text", i, 6);
    visualization_->drawArrow(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d(0.08, 0.18, 0.24), Vector4d(0.1, 0.9, 1.0, 1), "intent_speed_arrow", i, 6);
    visualization_->drawText(Eigen::Vector3d::Zero(), "", 0.32, Vector4d(0.1, 0.9, 1.0, 1), "intent_speed_text", i, 6);
  }
  // visualization_->drawSpheres({}, 0.2, Vector4d(0, 0, 1, 1), "refined_pts", 0, 6);
  // visualization_->drawLines({}, {}, 0.05, Vector4d(0.5, 0, 1, 1), "refined_view", 0, 6);
  // visualization_->drawLines({}, 0.07, Vector4d(0, 0, 1, 1), "refined_tour", 0, 6);
  visualization_->drawSpheres({}, 0.1, Vector4d(0, 0, 1, 1), "B-Spline", 0, 0);

  // visualization_->drawLines({}, {}, 0.03, Vector4d(1, 0, 0, 1), "current_pose", 0, 6);
}

void FastExplorationFSM::frontierCallback(const ros::TimerEvent& e) {
  if (state_ == WAIT_TRIGGER) {
    auto ft = expl_manager_->frontier_finder_;
    auto ed = expl_manager_->ed_;

    auto getColorVal = [&](const int& id, const int& num, const int& drone_id) {
      double a = (drone_id - 1) / double(num + 1);
      double b = 1 / double(num + 1);
      return a + b * double(id) / ed->frontiers_.size();
    };

    // ft->searchFrontiers();
    // ft->computeFrontiersToVisit();
    // ft->updateFrontierCostMatrix();

    // ft->getFrontiers(ed->frontiers_);
    // ft->getFrontierBoxes(ed->frontier_boxes_);

    expl_manager_->updateFrontierStruct(fd_->odom_pos_);

    cout << "odom: " << fd_->odom_pos_.transpose() << endl;
    vector<int> tmp_id1;
    vector<vector<int>> tmp_id2;
    bool status = expl_manager_->findGlobalTourOfGrid(
        { fd_->odom_pos_ }, { fd_->odom_vel_ }, tmp_id1, tmp_id2, true);

    // Draw frontier and bounding box
    for (int i = 0; i < ed->frontiers_.size(); ++i) {
      visualization_->drawCubes(ed->frontiers_[i], 0.1,
          visualization_->getColor(double(i) / ed->frontiers_.size(), 0.4), "frontier", i, 4);
      // getColorVal(i, expl_manager_->ep_->drone_num_, expl_manager_->ep_->drone_id_)
      // double(i) / ed->frontiers_.size()
      // visualization_->drawBox(ed->frontier_boxes_[i].first, ed->frontier_boxes_[i].second,
      // Vector4d(0.5, 0, 1, 0.3),
      //                         "frontier_boxes", i, 4);
    }
    for (int i = ed->frontiers_.size(); i < 50; ++i) {
      visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
      // visualization_->drawBox(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector4d(1, 0, 0, 0.3),
      // "frontier_boxes", i, 4);
    }
    if (status)
      visualize(2);
    else
      visualization_->drawLines({}, 0.07, Vector4d(0, 0.5, 0, 1), "grid_tour", 0, 6);

    // Draw grid tour
  }
}

void FastExplorationFSM::triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg) {

  // // Debug traj planner
  // Eigen::Vector3d pos;
  // pos << msg->pose.position.x, msg->pose.position.y, 1;
  // expl_manager_->ed_->next_pos_ = pos;

  // Eigen::Vector3d dir = pos - fd_->odom_pos_;
  // expl_manager_->ed_->next_yaw_ = atan2(dir[1], dir[0]);
  // fd_->go_back_ = true;
  // transitState(PLAN_TRAJ, "triggerCallback");
  // return;

  if (state_ != WAIT_TRIGGER) return;
  fd_->trigger_ = true;
  clearStopReason("exploration triggered");
  cout << "Triggered!" << endl;
  fd_->start_pos_ = fd_->odom_pos_;
  ROS_WARN_STREAM("Start expl pos: " << fd_->start_pos_.transpose());

  if (expl_manager_->updateFrontierStruct(fd_->odom_pos_) != 0) {
    transitState(PLAN_TRAJ, "triggerCallback");
  } else {
    updateStopReason("NO_FRONTIER_AT_TRIGGER", "No coverable frontier detected immediately after trigger");
    transitState(FINISH, "triggerCallback");
  }
}

void FastExplorationFSM::safetyCallback(const ros::TimerEvent& e) {
  if (state_ == EXPL_STATE::EXEC_TRAJ) {
    // Check safety and trigger replan if necessary
    double dist;
    bool safe = planner_manager_->checkTrajCollision(dist);
    if (!safe) {
      ROS_WARN("Replan: collision detected==================================");
      fd_->last_replan_trigger_ = std::string("safety collision detected, dist=") + std::to_string(dist);
      fd_->avoid_collision_ = true;
      transitState(PLAN_TRAJ, "safetyCallback");
    }
  }
}

void FastExplorationFSM::odometryCallback(const nav_msgs::OdometryConstPtr& msg) {
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_vel_(0) = msg->twist.twist.linear.x;
  fd_->odom_vel_(1) = msg->twist.twist.linear.y;
  fd_->odom_vel_(2) = msg->twist.twist.linear.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  if (!fd_->have_odom_) {
    fd_->have_odom_ = true;
    fd_->fsm_init_time_ = ros::Time::now();
  }
}

void FastExplorationFSM::transitState(EXPL_STATE new_state, string pos_call) {
  int pre_s = int(state_);
  state_ = new_state;
  ROS_INFO_STREAM("[" + pos_call + "]: Drone "
                  << getId()
                  << " from " + fd_->state_str_[pre_s] + " to " + fd_->state_str_[int(new_state)]);
  if (new_state == IDLE || new_state == FINISH) {
    logStopReason(std::string("STATE_TRANSIT_") + fd_->state_str_[int(new_state)]);
  }
}

void FastExplorationFSM::droneStateTimerCallback(const ros::TimerEvent& e) {
  // Broadcast own state periodically
  exploration_manager::DroneState msg;
  msg.drone_id = getId();

  auto& state = expl_manager_->ed_->swarm_state_[msg.drone_id - 1];

  if (fd_->static_state_) {
    state.pos_ = fd_->odom_pos_;
    state.vel_ = fd_->odom_vel_;
    state.yaw_ = fd_->odom_yaw_;
  } else {
    LocalTrajData* info = &planner_manager_->local_data_;
    double t_r = (ros::Time::now() - info->start_time_).toSec();
    state.pos_ = info->position_traj_.evaluateDeBoorT(t_r);
    state.vel_ = info->velocity_traj_.evaluateDeBoorT(t_r);
    state.yaw_ = info->yaw_traj_.evaluateDeBoorT(t_r)[0];
  }
  state.stamp_ = ros::Time::now().toSec();
  updateDroneIntentPrediction(state);
  msg.pos = { float(state.pos_[0]), float(state.pos_[1]), float(state.pos_[2]) };
  msg.vel = { float(state.vel_[0]), float(state.vel_[1]), float(state.vel_[2]) };
  msg.yaw = state.yaw_;
  for (auto id : state.grid_ids_) msg.grid_ids.push_back(id);
  msg.recent_attempt_time = state.recent_attempt_time_;
  msg.stamp = state.stamp_;
  msg.intent_valid = state.intent_valid_;
  msg.intent_dir = { float(state.intent_filt_dir_[0]), float(state.intent_filt_dir_[1]), float(state.intent_filt_dir_[2]) };
  msg.intent_conf = state.intent_conf_;
  msg.intent_sigma_rad = state.intent_sigma_rad_;
  msg.intent_t_int = state.intent_t_int_;
  msg.intent_aligned_speed = state.intent_aligned_speed_;
  msg.intent_aligned_speed_signed = state.intent_aligned_speed_signed_;
  msg.intent_speed_window = state.intent_speed_window_;
  msg.intent_update_stamp = state.intent_update_stamp_;
  msg.intent_pred_horizon = state.intent_pred_horizon_;
  msg.intent_progress_vel = { float(state.intent_progress_vel_[0]), float(state.intent_progress_vel_[1]), float(state.intent_progress_vel_[2]) };
  msg.intent_pred_pos = { float(state.intent_pred_pos_[0]), float(state.intent_pred_pos_[1]), float(state.intent_pred_pos_[2]) };

  drone_state_pub_.publish(msg);
}

void FastExplorationFSM::droneStateMsgCallback(const exploration_manager::DroneStateConstPtr& msg) {
  // Update other drones' states
  if (msg->drone_id == getId()) return;

  // Simulate swarm communication loss
  Eigen::Vector3d msg_pos(msg->pos[0], msg->pos[1], msg->pos[2]);
  if (!isInCommunicationRange(msg_pos)) return;

  auto& drone_state = expl_manager_->ed_->swarm_state_[msg->drone_id - 1];
  if (drone_state.stamp_ + 1e-4 >= msg->stamp) return;  // Avoid unordered msg

  drone_state.pos_ = Eigen::Vector3d(msg->pos[0], msg->pos[1], msg->pos[2]);
  drone_state.vel_ = Eigen::Vector3d(msg->vel[0], msg->vel[1], msg->vel[2]);
  drone_state.yaw_ = msg->yaw;
  drone_state.grid_ids_.clear();
  for (auto id : msg->grid_ids) drone_state.grid_ids_.push_back(id);
  drone_state.stamp_ = msg->stamp;
  drone_state.recent_attempt_time_ = msg->recent_attempt_time;
  updateDroneIntentPrediction(drone_state);
  if (msg->intent_valid && msg->intent_dir.size() >= 3) {
    drone_state.intent_valid_ = true;
    drone_state.intent_filt_dir_ = Eigen::Vector3d(msg->intent_dir[0], msg->intent_dir[1], msg->intent_dir[2]);
    normalizeSafe(drone_state.intent_filt_dir_);
    drone_state.intent_conf_ = msg->intent_conf;
    drone_state.intent_sigma_rad_ = msg->intent_sigma_rad;
    drone_state.intent_t_int_ = msg->intent_t_int;
    drone_state.intent_aligned_speed_ = msg->intent_aligned_speed;
    drone_state.intent_aligned_speed_signed_ = msg->intent_aligned_speed_signed;
    drone_state.intent_speed_window_ = msg->intent_speed_window;
    drone_state.intent_update_stamp_ = msg->intent_update_stamp;
    drone_state.intent_pred_horizon_ = msg->intent_pred_horizon;
    if (msg->intent_progress_vel.size() >= 3) {
      drone_state.intent_progress_vel_ = Eigen::Vector3d(
          msg->intent_progress_vel[0], msg->intent_progress_vel[1], msg->intent_progress_vel[2]);
    } else {
      drone_state.intent_progress_vel_ = drone_state.intent_filt_dir_ * drone_state.intent_aligned_speed_;
    }
    if (msg->intent_pred_pos.size() >= 3) {
      drone_state.intent_pred_pos_ = Eigen::Vector3d(
          msg->intent_pred_pos[0], msg->intent_pred_pos[1], msg->intent_pred_pos[2]);
    } else {
      drone_state.intent_pred_pos_ = drone_state.pos_ +
          drone_state.intent_progress_vel_ * drone_state.intent_pred_horizon_;
    }
  }

  // std::cout << "Drone " << getId() << " get drone " << int(msg->drone_id) << "'s state" <<
  // std::endl; std::cout << drone_state.pos_.transpose() << std::endl;
}

void FastExplorationFSM::optTimerCallback(const ros::TimerEvent& e) {
  if (!fp_->enable_pair_opt_) return;
  if (state_ == INIT) return;

  // Select nearby drone not interacting with recently
  auto& states = expl_manager_->ed_->swarm_state_;
  auto& state1 = states[getId() - 1];
  // bool urgent = (state1.grid_ids_.size() <= 1 /* && !state1.grid_ids_.empty() */);
  bool urgent = state1.grid_ids_.empty();
  auto tn = ros::Time::now().toSec();

  // Avoid frequent attempt
  if (tn - state1.recent_attempt_time_ < fp_->attempt_interval_) return;

  int select_id = -1;
  double max_interval = -1.0;
  for (int i = 0; i < states.size(); ++i) {
    if (i + 1 <= getId()) continue;
    // Check if have communication recently
    // or the drone just experience another opt
    // or the drone is interacted with recently /* !urgent &&  */
    // or the candidate drone dominates enough grids
    if (tn - states[i].stamp_ > 0.2) continue;
    if (tn - states[i].recent_attempt_time_ < fp_->attempt_interval_) continue;
    if (tn - states[i].recent_interact_time_ < fp_->pair_opt_interval_) continue;
    if (states[i].grid_ids_.size() + state1.grid_ids_.size() == 0) continue;

    double interval = tn - states[i].recent_interact_time_;
    if (interval <= max_interval) continue;
    select_id = i + 1;
    max_interval = interval;
  }
  if (select_id == -1) return;

  std::cout << "\nSelect: " << select_id << std::endl;
  ROS_WARN("Pair opt %d & %d", getId(), select_id);

  // Do pairwise optimization with selected drone, allocate the union of their domiance grids
  unordered_map<int, char> opt_ids_map;
  auto& state2 = states[select_id - 1];
  for (auto id : state1.grid_ids_) opt_ids_map[id] = 1;
  for (auto id : state2.grid_ids_) opt_ids_map[id] = 1;
  vector<int> opt_ids;
  for (auto pair : opt_ids_map) opt_ids.push_back(pair.first);

  std::cout << "Pair Opt id: ";
  for (auto id : opt_ids) std::cout << id << ", ";
  std::cout << "" << std::endl;

  // Find missed grids to reallocated them
  vector<int> actives, missed;
  expl_manager_->hgrid_->getActiveGrids(actives);
  findUnallocated(actives, missed);
  std::cout << "Missed: ";
  for (auto id : missed) std::cout << id << ", ";
  std::cout << "" << std::endl;
  opt_ids.insert(opt_ids.end(), missed.begin(), missed.end());

  // Do partition of the grid
  vector<Eigen::Vector3d> positions = { state1.pos_, state2.pos_ };
  vector<Eigen::Vector3d> velocities = { Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0) };
  vector<int> first_ids1, second_ids1, first_ids2, second_ids2;
  if (state_ != WAIT_TRIGGER) {
    expl_manager_->hgrid_->getConsistentGrid(
        state1.grid_ids_, state1.grid_ids_, first_ids1, second_ids1);
    expl_manager_->hgrid_->getConsistentGrid(
        state2.grid_ids_, state2.grid_ids_, first_ids2, second_ids2);
  }

  auto t1 = ros::Time::now();

  vector<int> ego_ids, other_ids;
  expl_manager_->allocateGrids(positions, velocities, { first_ids1, first_ids2 },
      { second_ids1, second_ids2 }, opt_ids, ego_ids, other_ids);

  double alloc_time = (ros::Time::now() - t1).toSec();

  std::cout << "Ego1  : ";
  for (auto id : state1.grid_ids_) std::cout << id << ", ";
  std::cout << "\nOther1: ";
  for (auto id : state2.grid_ids_) std::cout << id << ", ";
  std::cout << "\nEgo2  : ";
  for (auto id : ego_ids) std::cout << id << ", ";
  std::cout << "\nOther2: ";
  for (auto id : other_ids) std::cout << id << ", ";
  std::cout << "" << std::endl;

  // Check results
  double prev_app1 = expl_manager_->computeGridPathCost(state1.pos_, state1.grid_ids_, first_ids1,
      { first_ids1, first_ids2 }, { second_ids1, second_ids2 }, true);
  double prev_app2 = expl_manager_->computeGridPathCost(state2.pos_, state2.grid_ids_, first_ids2,
      { first_ids1, first_ids2 }, { second_ids1, second_ids2 }, true);
  std::cout << "prev cost: " << prev_app1 << ", " << prev_app2 << ", " << prev_app1 + prev_app2
            << std::endl;
  double cur_app1 = expl_manager_->computeGridPathCost(state1.pos_, ego_ids, first_ids1,
      { first_ids1, first_ids2 }, { second_ids1, second_ids2 }, true);
  double cur_app2 = expl_manager_->computeGridPathCost(state2.pos_, other_ids, first_ids2,
      { first_ids1, first_ids2 }, { second_ids1, second_ids2 }, true);
  std::cout << "cur cost : " << cur_app1 << ", " << cur_app2 << ", " << cur_app1 + cur_app2
            << std::endl;
  if (cur_app1 + cur_app2 > prev_app1 + prev_app2 + 0.1) {
    ROS_ERROR("Larger cost after reallocation");
    if (state_!=WAIT_TRIGGER) {
      return;
    }
  }

  if (!state1.grid_ids_.empty() && !ego_ids.empty() &&
      !expl_manager_->hgrid_->isConsistent(state1.grid_ids_[0], ego_ids[0])) {
    ROS_ERROR("Path 1 inconsistent");
  }
  if (!state2.grid_ids_.empty() && !other_ids.empty() &&
      !expl_manager_->hgrid_->isConsistent(state2.grid_ids_[0], other_ids[0])) {
    ROS_ERROR("Path 2 inconsistent");
  }

  // Update ego and other dominace grids
  auto last_ids2 = state2.grid_ids_;

  // Send the result to selected drone and wait for confirmation
  exploration_manager::PairOpt opt;
  opt.from_drone_id = getId();
  opt.to_drone_id = select_id;
  // opt.msg_type = 1;
  opt.stamp = tn;
  for (auto id : ego_ids) opt.ego_ids.push_back(id);
  for (auto id : other_ids) opt.other_ids.push_back(id);
  {
    Eigen::Vector3d sender_pos = getCurrentBroadcastPos();
    opt.sender_pos = { float(sender_pos[0]), float(sender_pos[1]), float(sender_pos[2]) };
  }

  for (int i = 0; i < fp_->repeat_send_num_; ++i) opt_pub_.publish(opt);

  ROS_WARN("Drone %d send opt request to %d, pair opt t: %lf, allocate t: %lf", getId(), select_id,
      ros::Time::now().toSec() - tn, alloc_time);

  // Reserve the result and wait...
  auto ed = expl_manager_->ed_;
  ed->ego_ids_ = ego_ids;
  ed->other_ids_ = other_ids;
  ed->pair_opt_stamp_ = opt.stamp;
  ed->wait_response_ = true;
  state1.recent_attempt_time_ = tn;
}

void FastExplorationFSM::findUnallocated(const vector<int>& actives, vector<int>& missed) {
  // Create map of all active
  unordered_map<int, char> active_map;
  for (auto ativ : actives) {
    active_map[ativ] = 1;
  }

  // Remove allocated ones
  for (auto state : expl_manager_->ed_->swarm_state_) {
    for (auto id : state.grid_ids_) {
      if (active_map.find(id) != active_map.end()) {
        active_map.erase(id);
      } else {
        // ROS_ERROR("Inactive grid %d is allocated.", id);
      }
    }
  }

  missed.clear();
  for (auto p : active_map) {
    missed.push_back(p.first);
  }
}

void FastExplorationFSM::optMsgCallback(const exploration_manager::PairOptConstPtr& msg) {
  if (msg->from_drone_id == getId() || msg->to_drone_id != getId()) return;

  // Check stamp to avoid unordered/repeated msg
  if (msg->stamp <= expl_manager_->ed_->pair_opt_stamps_[msg->from_drone_id - 1] + 1e-4) return;
  expl_manager_->ed_->pair_opt_stamps_[msg->from_drone_id - 1] = msg->stamp;

  if (msg->sender_pos.size() >= 3) {
    Eigen::Vector3d sender_pos(msg->sender_pos[0], msg->sender_pos[1], msg->sender_pos[2]);
    if (!isInCommunicationRange(sender_pos)) return;
  }

  auto& state1 = expl_manager_->ed_->swarm_state_[msg->from_drone_id - 1];
  auto& state2 = expl_manager_->ed_->swarm_state_[getId() - 1];

  // auto tn = ros::Time::now().toSec();
  exploration_manager::PairOptResponse response;
  response.from_drone_id = msg->to_drone_id;
  response.to_drone_id = msg->from_drone_id;
  response.stamp = msg->stamp;  // reply with the same stamp for verificaiton
  {
    Eigen::Vector3d sender_pos = getCurrentBroadcastPos();
    response.sender_pos = { float(sender_pos[0]), float(sender_pos[1]), float(sender_pos[2]) };
  }

  if (msg->stamp - state2.recent_attempt_time_ < fp_->attempt_interval_) {
    // Just made another pair opt attempt, should reject this attempt to avoid frequent changes
    ROS_WARN("Reject frequent attempt");
    response.status = 2;
  } else {
    // No opt attempt recently, and the grid info between drones are consistent, the pair opt
    // request can be accepted
    response.status = 1;

    // Update from the opt result
    state1.grid_ids_.clear();
    state2.grid_ids_.clear();
    for (auto id : msg->ego_ids) state1.grid_ids_.push_back(id);
    for (auto id : msg->other_ids) state2.grid_ids_.push_back(id);

    state1.recent_interact_time_ = msg->stamp;
    state2.recent_attempt_time_ = ros::Time::now().toSec();
    expl_manager_->ed_->reallocated_ = true;

    if (state_ == IDLE && !state2.grid_ids_.empty()) {
      clearStopReason("restart after pair optimization");
      transitState(PLAN_TRAJ, "optMsgCallback");
      ROS_WARN("Restart after opt!");
    }

    // if (!check_consistency(tmp1, tmp2)) {
    //   response.status = 2;
    //   ROS_WARN("Inconsistent grid info, reject pair opt");
    // } else {
    // }
  }
  for (int i = 0; i < fp_->repeat_send_num_; ++i) opt_res_pub_.publish(response);
}

void FastExplorationFSM::optResMsgCallback(
    const exploration_manager::PairOptResponseConstPtr& msg) {
  if (msg->from_drone_id == getId() || msg->to_drone_id != getId()) return;

  if (msg->sender_pos.size() >= 3) {
    Eigen::Vector3d sender_pos(msg->sender_pos[0], msg->sender_pos[1], msg->sender_pos[2]);
    if (!isInCommunicationRange(sender_pos)) return;
  }

  // Check stamp to avoid unordered/repeated msg
  if (msg->stamp <= expl_manager_->ed_->pair_opt_res_stamps_[msg->from_drone_id - 1] + 1e-4) return;
  expl_manager_->ed_->pair_opt_res_stamps_[msg->from_drone_id - 1] = msg->stamp;

  auto ed = expl_manager_->ed_;
  // Verify the consistency of pair opt via time stamp
  if (!ed->wait_response_ || fabs(ed->pair_opt_stamp_ - msg->stamp) > 1e-5) return;

  ed->wait_response_ = false;
  ROS_WARN("get response %d", int(msg->status));

  if (msg->status != 1) return;  // Receive 1 for valid opt

  auto& state1 = ed->swarm_state_[getId() - 1];
  auto& state2 = ed->swarm_state_[msg->from_drone_id - 1];
  state1.grid_ids_ = ed->ego_ids_;
  state2.grid_ids_ = ed->other_ids_;
  state2.recent_interact_time_ = ros::Time::now().toSec();
  ed->reallocated_ = true;

  if (state_ == IDLE && !state1.grid_ids_.empty()) {
    clearStopReason("restart after pair optimization response");
    transitState(PLAN_TRAJ, "optResMsgCallback");
    ROS_WARN("Restart after opt!");
  }
}

void FastExplorationFSM::swarmTrajCallback(const bspline::BsplineConstPtr& msg) {
  // Get newest trajs from other drones, for inter-drone collision avoidance
  auto& sdat = planner_manager_->swarm_traj_data_;

  // Ignore self trajectory
  if (msg->drone_id == sdat.drone_id_) return;

  if (msg->sender_pos.size() >= 3) {
    Eigen::Vector3d sender_pos(msg->sender_pos[0], msg->sender_pos[1], msg->sender_pos[2]);
    if (!isInCommunicationRange(sender_pos)) return;
  }

  // Ignore outdated trajectory
  if (sdat.receive_flags_[msg->drone_id - 1] == true &&
      msg->start_time.toSec() <= sdat.swarm_trajs_[msg->drone_id - 1].start_time_ + 1e-3)
    return;

  // Convert the msg to B-spline
  Eigen::MatrixXd pos_pts(msg->pos_pts.size(), 3);
  Eigen::VectorXd knots(msg->knots.size());
  for (int i = 0; i < msg->knots.size(); ++i) knots(i) = msg->knots[i];

  for (int i = 0; i < msg->pos_pts.size(); ++i) {
    pos_pts(i, 0) = msg->pos_pts[i].x;
    pos_pts(i, 1) = msg->pos_pts[i].y;
    pos_pts(i, 2) = msg->pos_pts[i].z;
  }

  // // Transform of drone's basecoor, optional step (skip if use swarm_pilot)
  // Eigen::Vector4d tf;
  // planner_manager_->edt_environment_->sdf_map_->getBaseCoor(msg->drone_id, tf);
  // double yaw = tf[3];
  // Eigen::Matrix3d rot;
  // rot << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1;
  // Eigen::Vector3d trans = tf.head<3>();
  // for (int i = 0; i < pos_pts.rows(); ++i) {
  //   Eigen::Vector3d tmp = pos_pts.row(i);
  //   tmp = rot * tmp + trans;
  //   pos_pts.row(i) = tmp;
  // }

  sdat.swarm_trajs_[msg->drone_id - 1].setUniformBspline(pos_pts, msg->order, 0.1);
  sdat.swarm_trajs_[msg->drone_id - 1].setKnot(knots);
  sdat.swarm_trajs_[msg->drone_id - 1].start_time_ = msg->start_time.toSec();
  sdat.receive_flags_[msg->drone_id - 1] = true;

  if (state_ == EXEC_TRAJ) {
    // Check collision with received trajectory
    if (!planner_manager_->checkSwarmCollision(msg->drone_id)) {
      ROS_ERROR("Drone %d collide with drone %d.", sdat.drone_id_, msg->drone_id);
      fd_->last_replan_trigger_ = std::string("swarm trajectory collision with drone ") + std::to_string(msg->drone_id);
      fd_->avoid_collision_ = true;
      transitState(PLAN_TRAJ, "swarmTrajCallback");
    }
  }
}

void FastExplorationFSM::swarmTrajTimerCallback(const ros::TimerEvent& e) {
  // Broadcast newest traj of this drone to others
  if (state_ == EXEC_TRAJ) {
    Eigen::Vector3d sender_pos = getCurrentBroadcastPos();
    fd_->newest_traj_.sender_pos = { float(sender_pos[0]), float(sender_pos[1]), float(sender_pos[2]) };
    swarm_traj_pub_.publish(fd_->newest_traj_);

  } else if (state_ == WAIT_TRIGGER) {
    // Publish a virtual traj at current pose, to avoid collision
    bspline::Bspline bspline;
    bspline.order = planner_manager_->pp_.bspline_degree_;
    bspline.start_time = ros::Time::now();
    bspline.traj_id = planner_manager_->local_data_.traj_id_;

    Eigen::MatrixXd pos_pts(4, 3);
    for (int i = 0; i < 4; ++i) pos_pts.row(i) = fd_->odom_pos_.transpose();

    for (int i = 0; i < pos_pts.rows(); ++i) {
      geometry_msgs::Point pt;
      pt.x = pos_pts(i, 0);
      pt.y = pos_pts(i, 1);
      pt.z = pos_pts(i, 2);
      bspline.pos_pts.push_back(pt);
    }

    NonUniformBspline tmp(pos_pts, planner_manager_->pp_.bspline_degree_, 1.0);
    Eigen::VectorXd knots = tmp.getKnot();
    for (int i = 0; i < knots.rows(); ++i) {
      bspline.knots.push_back(knots(i));
    }
    bspline.drone_id = expl_manager_->ep_->drone_id_;
    Eigen::Vector3d sender_pos = getCurrentBroadcastPos();
    bspline.sender_pos = { float(sender_pos[0]), float(sender_pos[1]), float(sender_pos[2]) };
    swarm_traj_pub_.publish(bspline);
  }
}

}  // namespace fast_planner
