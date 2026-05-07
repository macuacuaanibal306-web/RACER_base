#ifndef _EXPL_DATA_H_
#define _EXPL_DATA_H_

#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <ros/ros.h>
#include <bspline/Bspline.h>
#include <Eigen/StdVector>

using Eigen::Vector3d;
using std::string;
using std::vector;

namespace fast_planner {
struct FSMData {
  // FSM data
  bool trigger_, have_odom_, static_state_;
  vector<string> state_str_;
  string last_stop_reason_;
  string last_stop_detail_;
  string last_replan_trigger_;
  ros::Time last_stop_time_;
  ros::Time last_heartbeat_time_;

  Eigen::Vector3d odom_pos_, odom_vel_;  // odometry state
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_;

  Eigen::Vector3d start_pt_, start_vel_, start_acc_, start_yaw_;  // start state
  vector<Eigen::Vector3d> start_poss;
  bspline::Bspline newest_traj_;

  // Swarm collision avoidance
  bool avoid_collision_, go_back_;
  ros::Time fsm_init_time_;
  ros::Time last_check_frontier_time_;

  Eigen::Vector3d start_pos_;
};

struct FSMParam {
  double replan_thresh1_;
  double replan_thresh2_;
  double replan_thresh3_;
  double replan_time_;  // second

  // Swarm
  double attempt_interval_;   // Min interval of opt attempt
  double pair_opt_interval_;  // Min interval of successful pair opt
  int repeat_send_num_;
  bool enable_pair_opt_;
  bool enable_comm_constraint_;
  double comm_range_;
  bool enable_comm_status_vis_;
  double comm_status_timeout_;
  bool enable_refined_tour_vis_;
  bool enable_fast_greedy_tsp_;
  int greedy_tsp_max_nodes_;

  // Intent prediction / visualization
  bool enable_intent_prediction_vis_;
  bool show_all_intent_prediction_vis_;
  double intent_hist_max_age_;
  double intent_t_int_init_;
  double intent_t_int_min_;
  double intent_t_int_max_;
  double intent_goal_dist_min_;
  double intent_goal_dist_ref_;
  double intent_var_low_deg_;
  double intent_var_high_deg_;
  double intent_stable_deg_;
  double intent_stable_delta_deg_;
  double intent_shift_deg_;
  int intent_reset_count_thresh_;
  double intent_boundary_hit_time_thresh_;
  double intent_boundary_keep_recent_time_;
  int intent_boundary_keep_recent_samples_;
  double intent_boundary_min_speed_;
  double intent_hist_vector_length_;
  double intent_pred_line_length_;
  int intent_max_hist_vis_;
  double intent_speed_avg_time_;
  double intent_speed_min_dt_;
  double intent_pred_horizon_;
  bool enable_intent_speed_vis_;
  double intent_speed_vis_scale_;
  double intent_speed_vis_min_len_;
  double intent_speed_vis_max_len_;
};

struct DroneState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d pos_;
  Eigen::Vector3d vel_;
  double yaw_;
  double stamp_;                // Stamp of pos,vel,yaw
  double recent_attempt_time_;  // Stamp of latest opt attempt with any drone

  vector<int> grid_ids_;         // Id of grid tour
  double recent_interact_time_;  // Stamp of latest opt with this drone

  // Intent prediction state (local estimation only, not serialized directly)
  bool intent_valid_ = false;
  Eigen::Vector3d intent_goal_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d intent_raw_dir_ = Eigen::Vector3d::UnitX();
  Eigen::Vector3d intent_filt_dir_ = Eigen::Vector3d::UnitX();
  double intent_conf_ = 0.0;
  double intent_sigma_rad_ = 0.0;
  double intent_t_int_ = 4.0;
  double intent_aligned_speed_ = 0.0;        // m/s, mean forward progress along intent_filt_dir_
  double intent_aligned_speed_signed_ = 0.0; // m/s, signed projection diagnostic
  double intent_speed_window_ = 0.0;         // seconds of history used for speed averaging
  double intent_update_stamp_ = 0.0;
  double intent_pred_horizon_ = 3.0;
  Eigen::Vector3d intent_progress_vel_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d intent_pred_pos_ = Eigen::Vector3d::Zero();
  int intent_reset_count_ = 0;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> intent_hist_pos_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> intent_hist_raw_dir_;
  vector<double> intent_hist_stamp_;
  vector<double> intent_hist_dist_w_;
};

struct ExplorationData {
  vector<vector<Vector3d>> frontiers_;
  vector<vector<Vector3d>> dead_frontiers_;
  vector<pair<Vector3d, Vector3d>> frontier_boxes_;
  vector<Vector3d> points_;
  vector<Vector3d> averages_;
  vector<Vector3d> views_;
  vector<double> yaws_;
  vector<Vector3d> frontier_tour_;
  vector<vector<Vector3d>> other_tours_;

  vector<int> refined_ids_;
  vector<vector<Vector3d>> n_points_;
  vector<Vector3d> unrefined_points_;
  vector<Vector3d> refined_points_;
  vector<Vector3d> refined_views_;  // points + dir(yaw)
  vector<Vector3d> refined_views1_, refined_views2_;
  vector<Vector3d> refined_tour_;

  Vector3d next_goal_;
  vector<Vector3d> path_next_goal_, kino_path_;
  Vector3d next_pos_;
  double next_yaw_;

  // viewpoint planning
  // vector<Vector4d> views_;
  vector<Vector3d> views_vis1_, views_vis2_;
  vector<Vector3d> centers_, scales_;

  // Swarm, other drones' state
  vector<DroneState> swarm_state_;
  vector<double> pair_opt_stamps_, pair_opt_res_stamps_;
  vector<int> ego_ids_, other_ids_;
  double pair_opt_stamp_;
  bool reallocated_, wait_response_;

  // Coverage planning
  vector<Vector3d> grid_tour_, grid_tour2_;
  // int prev_first_id_;
  vector<int> last_grid_ids_;

  int plan_num_;
};

struct ExplorationParam {
  // params
  bool refine_local_;
  int refined_num_;
  double refined_radius_;
  int top_view_num_;
  double max_decay_;
  string tsp_dir_;   // resource dir of tsp solver
  string mtsp_dir_;  // resource dir of tsp solver
  double relax_time_;
  int init_plan_num_;

  // Visualization / search shortcuts
  bool enable_refined_tour_vis_;
  bool enable_fast_greedy_tsp_;
  int greedy_tsp_max_nodes_;

  // Region selection bias / inertia
  double dynamic_same_region_penalty_base_;
  double dynamic_same_region_exact_scale_;
  double dynamic_same_region_consistent_scale_;
  double dynamic_available_region_soft_min_;
  double dynamic_heading_penalty_;
  double dynamic_uturn_penalty_;
  double dynamic_heading_hard_cos_;
  double sensor_left_angle_;
  double sensor_right_angle_;
  double task_inertia_penalty_;
  double task_keep_bonus_;
  double task_inertia_available_scale_;

  // Swarm
  int drone_num_;
  int drone_id_;
};

}  // namespace fast_planner

#endif