// #include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <string>
#include <sstream>
#include <exploration_manager/fast_exploration_manager.h>
#include <thread>
#include <iostream>
#include <fstream>
#include <active_perception/graph_node.h>
#include <active_perception/graph_search.h>
#include <active_perception/perception_utils.h>
#include <active_perception/frontier_finder.h>
// #include <active_perception/uniform_grid.h>
#include <active_perception/hgrid.h>
#include <plan_env/raycast.h>
#include <plan_env/sdf_map.h>
#include <plan_env/edt_environment.h>
#include <plan_manage/planner_manager.h>
// #include <lkh_tsp_solver/lkh_interface.h>
// #include <lkh_mtsp_solver/lkh3_interface.h>
#include <lkh_tsp_solver/SolveTSP.h>
#include <lkh_mtsp_solver/SolveMTSP.h>

#include <exploration_manager/expl_data.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <visualization_msgs/Marker.h>
#include <cmath>

using namespace Eigen;

namespace fast_planner {
namespace {
std::string getDiagLogDirFM() {
  std::string dir;
  if (!ros::param::get("/diag/log_dir", dir) || dir.empty()) {
    const char* home = getenv("HOME");
    dir = std::string(home ? home : "/tmp") + "/.ros/swarm_diag_logs";
  }
  return dir;
}

bool getDiagFileLogEnableFM() {
  bool enable = true;
  ros::param::param<bool>("/diag/file_log_enable", enable, true);
  return enable;
}

void ensureDirRecursiveFM(const std::string& dir) {
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

void appendDiagLineFM(int drone_id, const std::string& tag, const std::string& line) {
  if (!getDiagFileLogEnableFM()) return;
  const std::string dir = getDiagLogDirFM();
  ensureDirRecursiveFM(dir);
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

template <typename T>
std::string vecToStrFM(const std::vector<T>& v) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) oss << ",";
    oss << v[i];
  }
  oss << "]";
  return oss.str();
}

template <typename T>
std::string vecVecToStrFM(const std::vector<std::vector<T>>& vv) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vv.size(); ++i) {
    if (i) oss << ";";
    oss << vecToStrFM(vv[i]);
  }
  oss << "]";
  return oss.str();
}

struct AdaptivePenaltyContext {
  double map_xy_diag = 20.0;
  double map_longest_span = 20.0;
  double map_xy_area = 400.0;
  double swarm_scale = 1.0;
  double same_region_penalty_floor = 10.0;
  double heading_penalty = 2.0;
  double uturn_penalty = 3.0;
  double task_inertia_penalty = 4.0;
  double task_keep_bonus = 2.0;
  double available_region_soft_min = 4.0;
  double scarcity_region_span = 8.0;
  double near_self_radius = 12.0;
  double isolated_start = 4.0;
  double isolated_span = 8.0;
  double isolated_fallback = 18.0;
  double continuation_accept_cost = 6.0;
  double continuation_group_radius = 6.0;
  double tiny_cell_floor = 6.0;
};

double clampRange(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

AdaptivePenaltyContext buildAdaptivePenaltyContext(const Eigen::Vector3d& map_size, int drone_num) {
  AdaptivePenaltyContext ctx;
  const double map_xy_diag = std::max(10.0, map_size.head<2>().norm());//地图对角线长度
  const double map_longest_span = std::max(8.0, std::max(map_size(0), map_size(1)));
  const double map_xy_area = std::max(64.0, map_size(0) * map_size(1));//地图平面面积
  const double swarm_scale = std::sqrt(std::max(1, drone_num));//编队尺度
  const double swarm_pressure = 1.0 + 0.18 * std::max(0, drone_num - 1);

  ctx.map_xy_diag = map_xy_diag;
  ctx.map_longest_span = map_longest_span;
  ctx.map_xy_area = map_xy_area;
  ctx.swarm_scale = swarm_scale;
  ctx.same_region_penalty_floor = 0.5 * map_xy_diag * swarm_pressure;//同区域惩罚下限
  ctx.heading_penalty = std::max(2.0, 0.03 * map_xy_diag * (0.80 + 0.12 * swarm_scale));//普通转向惩罚基准
  ctx.uturn_penalty = std::max(3.0, 0.06 * map_xy_diag * (0.80 + 0.08 * swarm_scale));//大角度掉头惩罚基准
  ctx.task_inertia_penalty = std::max(2.0, 0.12 * map_xy_diag * (0.90 + 0.10 * swarm_scale));//任务切换惩罚基准
  ctx.task_keep_bonus = std::max(1.0, 0.06 * map_xy_diag);//保持当前任务的奖励基准
  ctx.available_region_soft_min = std::max(swarm_scale, 2.0 + 1.2 * swarm_scale);//认为“候选区域已经够多”的阈值
  ctx.scarcity_region_span = std::max(8.0, 4.0 + 2.0 * swarm_scale);
  ctx.near_self_radius = clampRange(0.20 * map_xy_diag, 8.0, 0.33 * map_xy_diag);//小区域奖励的距离尺度
  ctx.isolated_start = clampRange(0.07 * map_xy_diag, 3.0, 0.15 * map_xy_diag);
  ctx.isolated_span = clampRange(0.14 * map_xy_diag, 5.0, 0.30 * map_xy_diag);
  ctx.isolated_fallback = clampRange(0.30 * map_xy_diag, 10.0, 0.45 * map_xy_diag);
  ctx.continuation_accept_cost = clampRange(0.10 * map_xy_diag, 4.0, 0.22 * map_xy_diag);
  ctx.continuation_group_radius = clampRange(0.10 * map_xy_diag, 4.0, 0.24 * map_xy_diag);
  ctx.tiny_cell_floor = std::max(6.0, 3.0 + 0.75 * drone_num);
  return ctx;
}

std::vector<int> greedyOrderFromMatrix(const Eigen::MatrixXd& mat, int drone_num) {
  std::vector<int> order;
  const int first_grid_node = 1 + drone_num;
  const int n = mat.rows();
  if (n <= first_grid_node) return order;
  std::vector<char> used(n, 0);
  int cur = 1;  // ego drone node
  while (true) {
    double best = 1e9;
    int best_j = -1;
    for (int j = first_grid_node; j < n; ++j) {
      if (used[j]) continue;
      if (mat(cur, j) < best) { best = mat(cur, j); best_j = j; }
    }
    if (best_j == -1) break;
    used[best_j] = 1;
    order.push_back(best_j);
    cur = best_j;
  }
  return order;
}

double clamp01(double v) {
  return std::max(0.0, std::min(1.0, v));
}

double wrapAnglePi(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}


struct IntentVectorContext {
  double candidate_span = M_PI;
  double local_radius = 10.0;
  double endpoint_scale = 4.0;
  double lateral_scale = 3.0;
  double overlap_band = 2.0;
  double penalty_cap = 10.0;
  double serial_bonus_cap = 4.0;
  double sample_step = 0.5;
};

struct IntentUnknownStats {
  double unknown_ratio = 0.0;
  double known_ratio = 0.0;
  double residual_unknown_ratio = 0.0;
};

double pointToAxisLateralDistance2D(const Eigen::Vector2d& point,
    const Eigen::Vector2d& origin, const Eigen::Vector2d& axis_dir) {
  const Eigen::Vector2d rel = point - origin;
  return (rel - rel.dot(axis_dir) * axis_dir).norm();
}

double pointToSegmentDistance2D(const Eigen::Vector2d& point,
    const Eigen::Vector2d& seg_start, const Eigen::Vector2d& seg_end) {
  const Eigen::Vector2d seg = seg_end - seg_start;
  const double seg_len2 = seg.squaredNorm();
  if (seg_len2 <= 1e-9) return (point - seg_start).norm();
  const double t = clampRange((point - seg_start).dot(seg) / seg_len2, 0.0, 1.0);
  return (point - (seg_start + t * seg)).norm();
}

double intervalOverlapRatio(double a0, double a1, double b0, double b1) {
  const double a_min = std::min(a0, a1);
  const double a_max = std::max(a0, a1);
  const double b_min = std::min(b0, b1);
  const double b_max = std::max(b0, b1);
  const double overlap = std::max(0.0, std::min(a_max, b_max) - std::max(a_min, b_min));
  const double denom = std::max(1e-3, std::min(a_max - a_min, b_max - b_min));
  return clamp01(overlap / denom);
}

IntentVectorContext buildIntentVectorContext(HGrid* hgrid, SDFMap* sdf_map,
    const AdaptivePenaltyContext& ctx, const int drone_num) {
  IntentVectorContext ictx;
  (void)hgrid;

  // Intention should only describe the next region to explore.
  // Do not infer an angular spread from all candidate regions or the full queued tour.
  // Use a single committed intention direction for the current candidate region.
  ictx.candidate_span = M_PI;

  const double map_resolution = (sdf_map != nullptr) ? std::max(1e-3, sdf_map->getResolution()) : 0.2;
  const double local_radius_from_area = std::sqrt(ctx.map_xy_area / std::max(1.0, double(drone_num)));
  ictx.local_radius = clampRange(local_radius_from_area,
      8.0 * map_resolution, 0.45 * ctx.map_xy_diag);
  ictx.endpoint_scale = clampRange(0.45 * ictx.local_radius,
      4.0 * map_resolution, 0.95 * ictx.local_radius);
  ictx.lateral_scale = clampRange(0.18 * ictx.local_radius,
      3.0 * map_resolution, 0.55 * ictx.local_radius);
  ictx.overlap_band = clampRange(0.5 * ictx.lateral_scale,
      2.0 * map_resolution, ictx.lateral_scale);

  const double direction_freedom = clamp01(ictx.candidate_span / M_PI);
  ictx.penalty_cap = 0.72 * ctx.map_xy_diag * (0.30 + 0.85 * direction_freedom) * ctx.swarm_scale;
  ictx.serial_bonus_cap = 0.22 * ctx.map_xy_diag * (0.20 + 0.80 * direction_freedom);
  ictx.sample_step = map_resolution;
  return ictx;
}

IntentUnknownStats sampleIntentUnknownStats(SDFMap* sdf_map,
    const Eigen::Vector3d& start, const Eigen::Vector3d& end,
    const Eigen::Vector2d& axis_dir, const double forward_cutoff,
    const Eigen::Vector3d& other_start, const Eigen::Vector3d& other_end,
    const IntentVectorContext& ictx) {
  IntentUnknownStats stats;
  if (sdf_map == nullptr) return stats;

  const Eigen::Vector3d seg = end - start;
  const double len = seg.norm();
  if (len < 1e-3) return stats;

  const int sample_num = std::max(2, int(std::ceil(len / std::max(1e-3, ictx.sample_step))) + 1);
  int in_map_cnt = 0;
  int unknown_cnt = 0;
  int known_cnt = 0;
  int residual_unknown_cnt = 0;
  const Eigen::Vector2d other_start2 = other_start.head<2>();
  const Eigen::Vector2d other_end2 = other_end.head<2>();

  for (int i = 0; i < sample_num; ++i) {
    const double t = (sample_num <= 1) ? 0.0 : double(i) / double(sample_num - 1);
    const Eigen::Vector3d pt = start + t * seg;
    if (!sdf_map->isInMap(pt)) continue;
    ++in_map_cnt;
    const int occ = sdf_map->getOccupancy(pt);
    if (occ == SDFMap::UNKNOWN) {
      ++unknown_cnt;
      const Eigen::Vector2d rel = (pt - start).head<2>();
      const double proj = rel.dot(axis_dir);
      const double dist_to_other = pointToSegmentDistance2D(pt.head<2>(), other_start2, other_end2);
      if (proj > forward_cutoff + ictx.overlap_band || dist_to_other > ictx.overlap_band) {
        ++residual_unknown_cnt;
      }
    } else if (occ == SDFMap::FREE) {
      ++known_cnt;
    }
  }

  if (in_map_cnt <= 0) return stats;
  stats.unknown_ratio = double(unknown_cnt) / double(in_map_cnt);
  stats.known_ratio = double(known_cnt) / double(in_map_cnt);
  stats.residual_unknown_ratio = double(residual_unknown_cnt) / double(in_map_cnt);
  return stats;
}

double computeIntentVectorConflictBias(const int& drone_idx, const Eigen::Vector3d& pos,
    const int& grid_id, const std::vector<int>& candidate_grid_ids,
    const std::vector<DroneState>& states, HGrid* hgrid, SDFMap* sdf_map,
    const AdaptivePenaltyContext& ctx, const int drone_num) {
  if (candidate_grid_ids.size() <= 1 || states.empty() || hgrid == nullptr) return 0.0;

  const Eigen::Vector3d cand_end = hgrid->getCenter(grid_id);
  Eigen::Vector2d cand_dir = (cand_end - pos).head<2>();
  const double cand_dist = cand_dir.norm();
  if (cand_dist < 1e-3) return 0.0;
  cand_dir /= cand_dist;

  const IntentVectorContext ictx = buildIntentVectorContext(
      hgrid, sdf_map, ctx, drone_num);
  if (ictx.penalty_cap <= 1e-3) return 0.0;

  double conflict_acc = 0.0;
  double serial_reward_acc = 0.0;

  for (int k = 0; k < (int)states.size(); ++k) {
    if (k == drone_idx) continue;
    if (states[k].grid_ids_.empty()) continue;

    const Eigen::Vector3d& other_pos = states[k].pos_;
    const Eigen::Vector3d other_end = hgrid->getCenter(states[k].grid_ids_.front());
    Eigen::Vector2d other_dir = (other_end - other_pos).head<2>();
    const double other_dist = other_dir.norm();
    if (other_dist < 1e-3) continue;
    other_dir /= other_dist;

    const double pair_start_dist = (other_pos - pos).head<2>().norm();
    const double pair_end_dist = (other_end - cand_end).head<2>().norm();
    if (pair_start_dist > ictx.local_radius && pair_end_dist > ictx.local_radius) continue;

    const double dot_dir = clampRange(cand_dir.dot(other_dir), -1.0, 1.0);
    const double same_dir = clamp01(dot_dir);
    const double opposite_dir = clamp01(-dot_dir);

    const double other_start_lateral = pointToAxisLateralDistance2D(other_pos.head<2>(), pos.head<2>(), cand_dir);
    const double other_end_lateral = pointToAxisLateralDistance2D(other_end.head<2>(), pos.head<2>(), cand_dir);
    const double lateral = 0.5 * (other_start_lateral + other_end_lateral);
    const double collinear = std::exp(-(lateral * lateral) /
        std::max(1e-6, ictx.lateral_scale * ictx.lateral_scale));
    if (collinear <= 1e-3) continue;

    const double proj_other_start = (other_pos - pos).head<2>().dot(cand_dir);
    const double proj_other_end = (other_end - pos).head<2>().dot(cand_dir);
    const double overlap = intervalOverlapRatio(0.0, cand_dist, proj_other_start, proj_other_end);

    const double endpoint_conflict_max = 0.8 * ctx.map_xy_diag;
    const double convoy_conflict_max = 0.65 * ctx.map_xy_diag;
    const double endpoint_proximity = std::exp(-(pair_end_dist * pair_end_dist) /
        std::max(1e-6, ictx.endpoint_scale * ictx.endpoint_scale));
    const double endpoint_conflict = endpoint_conflict_max * endpoint_proximity;//终点冲突代价上限随地图对角线自适应
    const double convoy_conflict = convoy_conflict_max * same_dir * collinear * overlap;//同向重复冲突上限随地图对角线自适应
    const double meeting_trigger = 1.0 - (1.0 - clamp01(overlap)) * (1.0 - endpoint_proximity);
    const double meeting_conflict = 2.45 * opposite_dir * collinear * meeting_trigger;//反向对冲冲突

    const double other_front = std::max(proj_other_start, proj_other_end);
    const double front_order = clamp01(other_front / std::max(cand_dist, ictx.sample_step));
    const double forward_extension = clamp01((cand_dist - other_front) /
        std::max(cand_dist, ictx.sample_step));
    const double serial_geom = same_dir * collinear * front_order * forward_extension;

    const IntentUnknownStats unknown_stats = sampleIntentUnknownStats(
        sdf_map, pos, cand_end, cand_dir, other_front, other_pos, other_end, ictx);
    const double serial_reward = serial_geom * clamp01(unknown_stats.residual_unknown_ratio);
    const double serial_waste = 1.20 * serial_geom *
        clamp01(unknown_stats.known_ratio - unknown_stats.residual_unknown_ratio);

    conflict_acc += endpoint_conflict + convoy_conflict + meeting_conflict + serial_waste;
    serial_reward_acc += serial_reward;
  }

  const double conflict_penalty = ictx.penalty_cap * (1.0 - std::exp(-conflict_acc));
  const double serial_bonus = ictx.serial_bonus_cap * (1.0 - std::exp(-serial_reward_acc));
  return conflict_penalty - serial_bonus;
}

double computeSmallRegionCleanupBonus(const double unknown_cells, const double total_cells,
    const double mean_unknown_cells, const int available_region_num,
    const AdaptivePenaltyContext& ctx) {
  if (unknown_cells <= 0.0 || mean_unknown_cells <= 1e-6) return 0.0;

  // Fewer remaining regions or larger teams => stronger cleanup preference for small leftover patches.
  const double scarcity = 1.0 - clamp01((double(available_region_num) - 2.0) /
      std::max(4.0, ctx.scarcity_region_span));
  const double relative_smallness = clamp01((mean_unknown_cells - unknown_cells) /
      std::max(1.0, mean_unknown_cells));
  const double tiny_threshold = std::max(ctx.tiny_cell_floor, 0.45 * mean_unknown_cells);
  const double tiny_fragment = clamp01((tiny_threshold - unknown_cells) /
      std::max(1.0, tiny_threshold));
  const double unknown_ratio = (total_cells > 1e-6) ? clamp01(unknown_cells / total_cells) : 1.0;
  const double low_unknown_ratio_small = (unknown_ratio < 0.20) ?
      clamp01((0.20 - unknown_ratio) / 0.20) : 0.0;

  // Negative value means lower score / lower cost => more likely to be selected.
  return -(0.30 + 1.05 * scarcity) *
         (0.18 + 0.45 * relative_smallness + 0.32 * tiny_fragment + 0.95 * low_unknown_ratio_small);
}

double computeNearbyIsolatedSmallRegionBonus(const Eigen::Vector3d& pos, const int gid,
    fast_planner::HGrid* hgrid, const std::vector<int>& grid_ids,
    const double mean_unknown_cells, const int available_region_num,
    const AdaptivePenaltyContext& ctx) {
  if (hgrid == nullptr || grid_ids.empty() || mean_unknown_cells <= 1e-6) return 0.0;

  const double unknown_cells = std::max(1, hgrid->getUnknownCellsNum(gid));
  const double total_cells = std::max(1, hgrid->getRegionTotalCellsNum(gid));
  const double unknown_ratio = clamp01(unknown_cells / total_cells);
  const double relative_smallness = clamp01((mean_unknown_cells - unknown_cells) /
      std::max(1.0, mean_unknown_cells));
  const double low_unknown_ratio_small = (unknown_ratio < 0.20) ?
      clamp01((0.20 - unknown_ratio) / 0.20) : 0.0;
  const double smallness_signal = std::max(relative_smallness, 0.95 * low_unknown_ratio_small);
  if (smallness_signal <= 0.05) return 0.0;

  const Eigen::Vector3d center = hgrid->getCenter(gid);
  const double self_dist = (center - pos).head<2>().norm();
  const double near_self = clamp01((ctx.near_self_radius - self_dist) /
      std::max(1.0, ctx.near_self_radius));
  if (near_self <= 0.0) return 0.0;

  const int gid_group = hgrid->getRegionGroupId(gid);
  double nearest_other = 1e9;
  for (int other_gid : grid_ids) {
    if (other_gid == gid) continue;
    if (gid_group >= 0 && hgrid->getRegionGroupId(other_gid) == gid_group) continue;
    const double d = (hgrid->getCenter(other_gid) - center).head<2>().norm();
    if (d < nearest_other) nearest_other = d;
  }
  if (nearest_other > 1e8) nearest_other = ctx.isolated_fallback;

  const double isolatedness = clamp01((nearest_other - ctx.isolated_start) /
      std::max(1.0, ctx.isolated_span));
  const double region_richness = clamp01((double(available_region_num) - 3.0) /
      std::max(4.0, 2.0 + 1.5 * ctx.swarm_scale));
  const double tiny_threshold = std::max(0.8 * ctx.tiny_cell_floor, 0.35 * mean_unknown_cells);
  const double tiny_fragment = clamp01((tiny_threshold - unknown_cells) /
      std::max(1.0, tiny_threshold));

  return -(1.10 + 1.20 * region_richness) * near_self *
         (0.40 * relative_smallness + 0.75 * isolatedness + 0.35 * tiny_fragment + 1.10 * low_unknown_ratio_small);
}

std::vector<int> greedyOrderFromSubset(const Eigen::MatrixXd& mat, const std::vector<int>& local_indices, int drone_num) {
  std::vector<int> order;
  if (local_indices.empty()) return order;
  std::vector<char> used(local_indices.size(), 0);
  int cur = 1;
  for (size_t step = 0; step < local_indices.size(); ++step) {
    double best = 1e9;
    int best_k = -1;
    for (size_t k = 0; k < local_indices.size(); ++k) {
      if (used[k]) continue;
      int node = 1 + drone_num + local_indices[k];
      if (mat(cur, node) < best) { best = mat(cur, node); best_k = (int)k; }
    }
    if (best_k < 0) break;
    used[best_k] = 1;
    order.push_back(local_indices[best_k]);
    cur = 1 + drone_num + local_indices[best_k];
  }
  return order;
}

void reorderIdsGreedy(const std::vector<Eigen::Vector3d>& positions, const std::vector<Eigen::Vector3d>& velocities,
    const std::vector<std::vector<int>>& first_ids, fast_planner::HGrid* hgrid, std::vector<int>& ids) {
  if (ids.size() <= 1 || hgrid == nullptr) return;
  Eigen::MatrixXd mat;
  hgrid->getCostMatrix(positions, velocities, first_ids, std::vector<std::vector<int>>(first_ids.size()), ids, mat);
  auto local = greedyOrderFromMatrix(mat, positions.size());
  std::vector<int> new_ids;
  for (auto node : local) {
    int idx = node - 1 - (int)positions.size();
    if (idx >= 0 && idx < (int)ids.size()) new_ids.push_back(ids[idx]);
  }
  if (new_ids.size() == ids.size()) ids.swap(new_ids);
}

std::vector<int> inferSpatialContinuationIds(const Eigen::Vector3d& pos,
    const Eigen::Vector3d& vel, fast_planner::HGrid* hgrid, const std::vector<int>& grid_ids,
    const AdaptivePenaltyContext& ctx) {
  std::vector<int> first_ids;
  if (hgrid == nullptr || !hgrid->isTaskRegionMode() || grid_ids.empty()) return first_ids;

  Eigen::Vector2d vel_dir = vel.head<2>();
  const double vel_norm = vel_dir.norm();
  if (vel_norm > 1e-3) vel_dir /= vel_norm;

  double mean_unknown_cells = 0.0;
  for (int gid : grid_ids) mean_unknown_cells += std::max(1, hgrid->getUnknownCellsNum(gid));
  mean_unknown_cells /= std::max<size_t>(1, grid_ids.size());

  double best_score = 1e9, second_score = 1e9;
  int best_gid = -1;
  for (int gid : grid_ids) {
    Eigen::Vector3d center = hgrid->getCenter(gid);
    Eigen::Vector2d diff = (center - pos).head<2>();
    const double dist = diff.norm();
    double score = dist;
    if (vel_norm > 0.2 && dist > 1e-3) {
      diff /= dist;
      const double align = vel_dir.dot(diff);
      score += 0.8 * (1.0 - align);
      if (align < 0.0) score += 1.2 * (-align);
    }
    score += 0.40 * computeSmallRegionCleanupBonus(
        hgrid->getUnknownCellsNum(gid), hgrid->getRegionTotalCellsNum(gid),
        mean_unknown_cells, (int)grid_ids.size(), ctx);
    score += computeNearbyIsolatedSmallRegionBonus(
        pos, gid, hgrid, grid_ids, mean_unknown_cells, (int)grid_ids.size(), ctx);
    if (score < best_score) {
      second_score = best_score;
      best_score = score;
      best_gid = gid;
    } else if (score < second_score) {
      second_score = score;
    }
  }

  if (best_gid < 0) return first_ids;

  bool continuation_available = (best_score < ctx.continuation_accept_cost);
  if (!continuation_available && second_score < 1e8) {
    continuation_available = (best_score < 0.72 * second_score);
  }
  if (!continuation_available) return first_ids;

  first_ids.push_back(best_gid);
  const int best_group = hgrid->getRegionGroupId(best_gid);
  if (best_group >= 0) {
    std::vector<std::pair<double, int>> same_group;
    const Eigen::Vector3d best_center = hgrid->getCenter(best_gid);
    for (int gid : grid_ids) {
      if (gid == best_gid) continue;
      if (hgrid->getRegionGroupId(gid) != best_group) continue;
      double d = (hgrid->getCenter(gid) - best_center).head<2>().norm();
      if (d <= ctx.continuation_group_radius) same_group.push_back({d, gid});
    }
    std::sort(same_group.begin(), same_group.end(),
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
          return a.first < b.first;
        });
    for (const auto& item : same_group) first_ids.push_back(item.second);
  }
  return first_ids;
}

}  // namespace
// SECTION interfaces for setup and query

FastExplorationManager::FastExplorationManager() {
}

FastExplorationManager::~FastExplorationManager() {
  ViewNode::astar_.reset();
  ViewNode::caster_.reset();
  ViewNode::map_.reset();
}
//初始化
void FastExplorationManager::initialize(ros::NodeHandle& nh) {
  planner_manager_.reset(new FastPlannerManager);//初始化路径规划
  planner_manager_->initPlanModules(nh);

  edt_environment_ = planner_manager_->edt_environment_;//获取环境
  sdf_map_ = edt_environment_->sdf_map_;//获取地图
  frontier_finder_.reset(new FrontierFinder(edt_environment_, nh));//初始化前沿检测器
  // uniform_grid_.reset(new UniformGrid(edt_environment_, nh));
  hgrid_.reset(new HGrid(edt_environment_, nh));//初始化网格管理器
  // view_finder_.reset(new ViewFinder(edt_environment_, nh));

  ed_.reset(new ExplorationData);//探索任务数据
  ep_.reset(new ExplorationParam);//探索任务参数
  //读取ros参数
  nh.param("exploration/refine_local", ep_->refine_local_, true);
  nh.param("exploration/refined_num", ep_->refined_num_, -1);
  nh.param("exploration/refined_radius", ep_->refined_radius_, -1.0);
  nh.param("exploration/top_view_num", ep_->top_view_num_, -1);
  nh.param("exploration/max_decay", ep_->max_decay_, -1.0);
  nh.param("exploration/tsp_dir", ep_->tsp_dir_, string("null"));
  nh.param("exploration/mtsp_dir", ep_->mtsp_dir_, string("null"));
  nh.param("exploration/relax_time", ep_->relax_time_, 1.0);
  nh.param("exploration/drone_num", ep_->drone_num_, 1);
  nh.param("exploration/drone_id", ep_->drone_id_, 1);
  nh.param("exploration/init_plan_num", ep_->init_plan_num_, 2);
  nh.param("manager/enable_refined_tour_vis", ep_->enable_refined_tour_vis_, false);
  nh.param("manager/enable_fast_greedy_tsp", ep_->enable_fast_greedy_tsp_, true);
  nh.param("manager/greedy_tsp_max_nodes", ep_->greedy_tsp_max_nodes_, 18);
  nh.param("manager/dynamic_same_region_penalty_base", ep_->dynamic_same_region_penalty_base_, 8.0);
  nh.param("manager/dynamic_same_region_exact_scale", ep_->dynamic_same_region_exact_scale_, 1.0);
  nh.param("manager/dynamic_same_region_consistent_scale", ep_->dynamic_same_region_consistent_scale_, 0.5);
  nh.param("manager/dynamic_available_region_soft_min", ep_->dynamic_available_region_soft_min_, 4.0);
  nh.param("manager/dynamic_heading_penalty", ep_->dynamic_heading_penalty_, 4.0);
  nh.param("manager/dynamic_uturn_penalty", ep_->dynamic_uturn_penalty_, 20.0);
  nh.param("manager/dynamic_heading_hard_cos", ep_->dynamic_heading_hard_cos_, 0.2);
  nh.param("perception_utils/left_angle", ep_->sensor_left_angle_, 0.69222);
  nh.param("perception_utils/right_angle", ep_->sensor_right_angle_, 0.68901);
  nh.param("manager/task_inertia_penalty", ep_->task_inertia_penalty_, 8.0);
  nh.param("manager/task_keep_bonus", ep_->task_keep_bonus_, 4.0);
  nh.param("manager/task_inertia_available_scale", ep_->task_inertia_available_scale_, 1.0);

  ed_->swarm_state_.resize(ep_->drone_num_);
  ed_->pair_opt_stamps_.resize(ep_->drone_num_);
  ed_->pair_opt_res_stamps_.resize(ep_->drone_num_);
  for (int i = 0; i < ep_->drone_num_; ++i) {
    ed_->swarm_state_[i].stamp_ = 0.0;
    ed_->pair_opt_stamps_[i] = 0.0;
    ed_->pair_opt_res_stamps_[i] = 0.0;
  }
  planner_manager_->swarm_traj_data_.init(ep_->drone_id_, ep_->drone_num_);

  nh.param("exploration/vm", ViewNode::vm_, -1.0);
  nh.param("exploration/am", ViewNode::am_, -1.0);
  nh.param("exploration/yd", ViewNode::yd_, -1.0);
  nh.param("exploration/ydd", ViewNode::ydd_, -1.0);
  nh.param("exploration/w_dir", ViewNode::w_dir_, -1.0);

  ViewNode::astar_.reset(new Astar);
  ViewNode::astar_->init(nh, edt_environment_);
  ViewNode::map_ = sdf_map_;

  double resolution_ = sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  sdf_map_->getRegion(origin, size);
  ViewNode::caster_.reset(new RayCaster);
  ViewNode::caster_->setParams(resolution_, origin);

  planner_manager_->path_finder_->lambda_heu_ = 1.0;
  // planner_manager_->path_finder_->max_search_time_ = 0.05;
  planner_manager_->path_finder_->max_search_time_ = 1.0;

  tsp_client_ =
      nh.serviceClient<lkh_mtsp_solver::SolveMTSP>("/solve_tsp_" + to_string(ep_->drone_id_), true);
  acvrp_client_ = nh.serviceClient<lkh_mtsp_solver::SolveMTSP>(
      "/solve_acvrp_" + to_string(ep_->drone_id_), true);

  // Swarm
  for (auto& state : ed_->swarm_state_) {
    state.stamp_ = 0.0;
    state.recent_interact_time_ = 0.0;
    state.recent_attempt_time_ = 0.0;
  }
  ed_->last_grid_ids_ = {};
  ed_->reallocated_ = true;
  ed_->pair_opt_stamp_ = 0.0;
  ed_->wait_response_ = false;
  ed_->plan_num_ = 0;

  // Analysis
  // ofstream fout;
  // fout.open("/home/boboyu/Desktop/RAL_Time/frontier.txt");
  // fout.close();
}
//探索任务主逻辑：任务规划、全局和局部路径规划、调用planTrajToView进行路径规划
int FastExplorationManager::planExploreMotion(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw) {
  ros::Time t1 = ros::Time::now();
  auto t2 = t1;//记录当前时间

  std::cout << "start pos: " << pos.transpose() << ", vel: " << vel.transpose()
            << ", acc: " << acc.transpose() << std::endl;

  // Do global and local tour planning and retrieve the next viewpoint

  ed_->frontier_tour_.clear();//清除旧的前沿路径
  Vector3d next_pos;//下一个位资
  double next_yaw;//下一个偏航角
  // Find the tour passing through viewpoints
  // Optimal tour is returned as indices of frontier
  vector<int> grid_ids, frontier_ids;
  // findGlobalTour(pos, vel, yaw, indices);
  findGridAndFrontierPath(pos, vel, yaw, grid_ids, frontier_ids);//为四旋翼无人机集群寻找最优协同巡游路径
  {
    std::ostringstream oss;
    oss << "planExploreMotion pos=[" << pos.transpose() << "]"
        << ", vel=[" << vel.transpose() << "]"
        << ", yaw=[" << yaw.transpose() << "]"
        << ", assigned_grid_ids=" << vecToStrFM(grid_ids)
        << ", frontier_ids=" << vecToStrFM(frontier_ids)
        << ", frontier_count=" << frontier_ids.size();
    appendDiagLineFM(ep_->drone_id_, "TASK_ASSIGN", oss.str());
  }
  //处理无网格情况
  if (grid_ids.empty()) {

    // No grid is assigned to this drone, but keep moving is necessary
    ROS_WARN("Empty grid");

    double min_cost = 100000;
    int min_cost_id = -1;
    vector<Vector3d> tmp_path;
    for (int i = 0; i < (int)ed_->points_.size(); ++i) {
      auto tmp_cost = ViewNode::computeCost(pos, ed_->points_[i], yaw[0], ed_->yaws_[i], vel, yaw[1], tmp_path);
      if (tmp_cost < min_cost) {
        min_cost = tmp_cost;
        min_cost_id = i;
      }
    }
    if (min_cost_id < 0) return FAIL;
    next_pos = ed_->points_[min_cost_id];
    next_yaw = ed_->yaws_[min_cost_id];

  } else if (frontier_ids.size() == 0) {//如果当前网格中没有前沿
    // // 已分配的网格不包含前沿，寻找最接近该网格的前沿
    // ROS_WARN("No frontier in grid");
    //获取当前网格中心点
    Eigen::Vector3d grid_center = ed_->grid_tour_[1];
    //初始化变量
    double min_cost = 100000;
    int min_cost_id = -1;
    for (int i = 0; i < ed_->points_.size(); ++i) {//遍历实际坐标点
      // double cost = (grid_center - ed_->averages_[i]).norm();
      vector<Eigen::Vector3d> path;
      double cost = ViewNode::computeCost(//计算每个目标点到网格中心的成本
          grid_center, ed_->averages_[i], 0, 0, Eigen::Vector3d(0, 0, 0), 0, path);
      if (cost < min_cost) {
        min_cost = cost;
        min_cost_id = i;
      }
    }
    next_pos = ed_->points_[min_cost_id];
    next_yaw = ed_->yaws_[min_cost_id];

    // // Simply go to the center of the unknown grid
    // next_pos = grid_center;
    // Eigen::Vector3d dir = grid_center - pos;
    // next_yaw = atan2(dir[1], dir[0]);

  } else if (frontier_ids.size() == 1) {
    // ROS_WARN("Single frontier");
    if (ep_->refine_local_) {//是否使用局部优化
      // 单一边界，为其寻找最小成本观测点
      ed_->refined_ids_ = { frontier_ids[0] };//储存当前前沿id
      ed_->unrefined_points_ = { ed_->points_[frontier_ids[0]] };//存储当前前沿原始点
      ed_->n_points_.clear();//清空观测点
      vector<vector<double>> n_yaws;//存储观测点偏航角
      frontier_finder_->getViewpointsInfo(//获取当前前沿观测点信息
          pos, { frontier_ids[0] }, ep_->top_view_num_, ep_->max_decay_, ed_->n_points_, n_yaws);

      if (grid_ids.size() <= 1) {
        // 只有一个网格被分配
        double min_cost = 100000;
        int min_cost_id = -1;
        vector<Vector3d> tmp_path;
        for (int i = 0; i < ed_->n_points_[0].size(); ++i) {
          auto tmp_cost = ViewNode::computeCost(//遍历观测点计算成本
              pos, ed_->n_points_[0][i], yaw[0], n_yaws[0][i], vel, yaw[1], tmp_path);
          if (tmp_cost < min_cost) {
            min_cost = tmp_cost;
            min_cost_id = i;
          }
        }
        next_pos = ed_->n_points_[0][min_cost_id];
        next_yaw = n_yaws[0][min_cost_id];
      } else {
        // 存在多个网格，将下一个网格纳入路径规划考虑
        // vector<Eigen::Vector3d> grid_pos = { ed_->grid_tour_[2] };
        // Eigen::Vector3d dir = ed_->grid_tour_[2] - ed_->grid_tour_[1];
        // vector<double> grid_yaw = { atan2(dir[1], dir[0]) };

        Eigen::Vector3d grid_pos;//网格位置
        double grid_yaw;//网格偏航角
        if (hgrid_->getNextGrid(grid_ids, grid_pos, grid_yaw)) {//如果获取到了下一个网格的位置和偏航角
          ed_->n_points_.push_back({ grid_pos });//下一网格位置
          n_yaws.push_back({ grid_yaw });//下一网格偏航角
        }
        //优化局部路径
        ed_->refined_points_.clear();
        ed_->refined_views_.clear();
        vector<double> refined_yaws;
        refineLocalTour(pos, vel, yaw, ed_->n_points_, n_yaws, ed_->refined_points_, refined_yaws);
        next_pos = ed_->refined_points_[0];
        next_yaw = refined_yaws[0];
      }
      ed_->refined_points_ = { next_pos };
      ed_->refined_views_ = { next_pos + 2.0 * Vector3d(cos(next_yaw), sin(next_yaw), 0) };//计算观测方向
    }
  } else {
    // ROS_WARN("Multiple frontier");
    // 已分配超过两个边界
    // 对全局路线中的后续几个观测点进行优化
    t1 = ros::Time::now();
    //初始化
    ed_->refined_ids_.clear();
    ed_->unrefined_points_.clear();
    int knum = min(int(frontier_ids.size()), ep_->refined_num_);//将要优化的前沿数量
    for (int i = 0; i < knum; ++i) {//选择前沿中前knum个点
      auto tmp = ed_->points_[frontier_ids[i]];
      ed_->unrefined_points_.push_back(tmp);
      ed_->refined_ids_.push_back(frontier_ids[i]);
      if ((tmp - pos).norm() > ep_->refined_radius_ && ed_->refined_ids_.size() >= 2) break;
    }//如果某个前沿点与当前位置的距离大于 ep_->refined_radius_，并且已经选择了至少 2 个前沿点，则停止选择

    // 为接下来的K个边界获取前N个观测点
    ed_->n_points_.clear();
    vector<vector<double>> n_yaws;
    frontier_finder_->getViewpointsInfo(
        pos, ed_->refined_ids_, ep_->top_view_num_, ep_->max_decay_, ed_->n_points_, n_yaws);
    //选择优化后的第一个点作为下一步的目标  
    ed_->refined_points_.clear();
    ed_->refined_views_.clear();
    vector<double> refined_yaws;
    refineLocalTour(pos, vel, yaw, ed_->n_points_, n_yaws, ed_->refined_points_, refined_yaws);
    next_pos = ed_->refined_points_[0];
    next_yaw = refined_yaws[0];

    // 生成并获取视图可视化的标记点
    for (int i = 0; i < ed_->refined_points_.size(); ++i) {
      Vector3d view =
          ed_->refined_points_[i] + 2.0 * Vector3d(cos(refined_yaws[i]), sin(refined_yaws[i]), 0);
      ed_->refined_views_.push_back(view);
    }
    ed_->refined_views1_.clear();
    ed_->refined_views2_.clear();
    for (int i = 0; i < ed_->refined_points_.size(); ++i) {//生成视野范围的可视化标记
      vector<Vector3d> v1, v2;
      frontier_finder_->percep_utils_->setPose(ed_->refined_points_[i], refined_yaws[i]);
      frontier_finder_->percep_utils_->getFOV(v1, v2);
      ed_->refined_views1_.insert(ed_->refined_views1_.end(), v1.begin(), v1.end());
      ed_->refined_views2_.insert(ed_->refined_views2_.end(), v2.begin(), v2.end());
    }
    double local_time = (ros::Time::now() - t1).toSec();
    ROS_INFO("Local refine time: %lf", local_time);
  }

  std::cout << "Next view: " << next_pos.transpose() << ", " << next_yaw << std::endl;
  ed_->next_pos_ = next_pos;
  ed_->next_yaw_ = next_yaw;

  if (planTrajToView(pos, vel, acc, yaw, next_pos, next_yaw) == FAIL) {
    return FAIL;
  }

  double total = (ros::Time::now() - t2).toSec();
  ROS_INFO("Total time: %lf", total);
  ROS_WARN_COND(total > 0.25, "Total time too long!!!");
  {
    std::ostringstream oss;
    oss << "planExploreMotion result=SUCCEED"
        << ", next_pos=[" << next_pos.transpose() << "]"
        << ", next_yaw=" << next_yaw
        << ", grid_ids=" << vecToStrFM(grid_ids)
        << ", frontier_ids=" << vecToStrFM(frontier_ids)
        << ", path_next_goal_size=" << ed_->path_next_goal_.size();
    appendDiagLineFM(ep_->drone_id_, "TASK_ASSIGN", oss.str());
  }

  return SUCCEED;
}
//路径规划：轨迹规划，生成轨迹
int FastExplorationManager::planTrajToView(const Vector3d& pos, const Vector3d& vel,
    const Vector3d& acc, const Vector3d& yaw, const Vector3d& next_pos, const double& next_yaw) {

  // 规划到下一个观测点的轨迹（位置和偏航角）
  auto t1 = ros::Time::now();

  // 计算偏航角的时间下限并用于轨迹生成
  double diff0 = next_yaw - yaw[0];//偏航差值
  double diff1 = fabs(diff0);//绝对值
  double time_lb = min(diff1, 2 * M_PI - diff1) / ViewNode::yd_;//最小偏航角的变化时间

  // 生成x、y、z的轨迹
  bool goal_unknown = (edt_environment_->sdf_map_->getOccupancy(next_pos) == SDFMap::UNKNOWN);//目标点是否在未知区域
  // bool start_unknown = (edt_environment_->sdf_map_->getOccupancy(pos) == SDFMap::UNKNOWN);
  bool optimistic = ed_->plan_num_ < ep_->init_plan_num_;//如果当前已经完成小于初始化设定的最大规划次数，则使用乐观估计，积极探索路径
  planner_manager_->path_finder_->reset();//重置路径搜索
  if (planner_manager_->path_finder_->search(pos, next_pos, optimistic) != Astar::REACH_END) {
    ROS_ERROR("No path to next viewpoint");
    return FAIL;
  }
  ed_->path_next_goal_ = planner_manager_->path_finder_->getPath();//获取搜索到的路径
  shortenPath(ed_->path_next_goal_);//优化路径，减少不必要的点
  ed_->kino_path_.clear();

  const double radius_far = 7.0;
  const double radius_close = 1.5;
  const double len = Astar::pathLength(ed_->path_next_goal_);
  if (len < radius_close || optimistic) {
    // 下一个观测点非常接近，无需搜索动力学路径，直接使用基于航点的方法
    // optimization
    planner_manager_->planExploreTraj(ed_->path_next_goal_, vel, acc, time_lb);
    ed_->next_goal_ = next_pos;
    // std::cout << "Close goal." << std::endl;
    if (ed_->plan_num_ < ep_->init_plan_num_) {
      ed_->plan_num_++;
      ROS_WARN("init plan.");
    }
  } else if (len > radius_far) {
    // 下一个观测点距离较远，在几何路径上选择中间目标点（此举同时解决死胡同问题）
    // dead end)
    std::cout << "Far goal." << std::endl;
    double len2 = 0.0;
    vector<Eigen::Vector3d> truncated_path = { ed_->path_next_goal_.front() };
    for (int i = 1; i < ed_->path_next_goal_.size() && len2 < radius_far; ++i) {
      auto cur_pt = ed_->path_next_goal_[i];
      len2 += (cur_pt - truncated_path.back()).norm();
      truncated_path.push_back(cur_pt);
    }
    ed_->next_goal_ = truncated_path.back();
    planner_manager_->planExploreTraj(truncated_path, vel, acc, time_lb);
  } else {
    // 搜索精确到达下一个观测点的动力学路径并进行优化
    std::cout << "Mid goal" << std::endl;
    ed_->next_goal_ = next_pos;

    if (!planner_manager_->kinodynamicReplan(
            pos, vel, acc, ed_->next_goal_, Vector3d(0, 0, 0), time_lb))
      return FAIL;
    ed_->kino_path_ = planner_manager_->kino_path_finder_->getKinoTraj(0.02);
  }

  if (planner_manager_->local_data_.position_traj_.getTimeSum() < time_lb - 0.5)
    ROS_ERROR("Lower bound not satified!");

  double traj_plan_time = (ros::Time::now() - t1).toSec();

  t1 = ros::Time::now();
  planner_manager_->planYawExplore(yaw, next_yaw, true, ep_->relax_time_);
  double yaw_time = (ros::Time::now() - t1).toSec();
  ROS_INFO("Traj: %lf, yaw: %lf", traj_plan_time, yaw_time);

  return SUCCEED;
}
//前沿检测和路径规划：前沿检测，路径规划，生成网格路径和前沿路径
int FastExplorationManager::updateFrontierStruct(const Eigen::Vector3d& pos) {

  auto t1 = ros::Time::now();
  auto t2 = t1;
  ed_->views_.clear();

  // 搜索边界并将其分组为簇
  frontier_finder_->searchFrontiers();

  double frontier_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // 为所有簇寻找观测点（x,y,z,yaw）；找出信息量丰富的观测点
  frontier_finder_->computeFrontiersToVisit();

  // 检索更新后的信息
  frontier_finder_->getFrontiers(ed_->frontiers_);
  frontier_finder_->getDormantFrontiers(ed_->dead_frontiers_);
  frontier_finder_->getFrontierBoxes(ed_->frontier_boxes_);

  frontier_finder_->getTopViewpointsInfo(pos, ed_->points_, ed_->yaws_, ed_->averages_);
  for (int i = 0; i < ed_->points_.size(); ++i)
    ed_->views_.push_back(
        ed_->points_[i] + 2.0 * Vector3d(cos(ed_->yaws_[i]), sin(ed_->yaws_[i]), 0));

  if (ed_->frontiers_.empty()) {
    ROS_WARN_STREAM("No coverable frontier. pos=[" << pos.transpose() << "] views=" << ed_->views_.size()
                    << " points=" << ed_->points_.size() << " dead_frontiers=" << ed_->dead_frontiers_.size());
    return 0;
  }

  double view_time = (ros::Time::now() - t1).toSec();

  t1 = ros::Time::now();
  frontier_finder_->updateFrontierCostMatrix();

  double mat_time = (ros::Time::now() - t1).toSec();
  double total_time = frontier_time + view_time + mat_time;
  ROS_INFO("Drone %d: frontier t: %lf, viewpoint t: %lf, mat: %lf", ep_->drone_id_, frontier_time,
      view_time, mat_time);

  ROS_INFO("Total t: %lf", (ros::Time::now() - t2).toSec());
  return ed_->frontiers_.size();
}
//先按全局网格分配任务，再在分配的网格内找前沿并规划路径
void FastExplorationManager::findGridAndFrontierPath(const Vector3d& cur_pos,
    const Vector3d& cur_vel, const Vector3d& cur_yaw, vector<int>& grid_ids,
    vector<int>& frontier_ids) {
  auto t1 = ros::Time::now();

  // 根据状态时间戳选择附近无人机
  vector<Eigen::Vector3d> positions = { cur_pos };
  // vector<Eigen::Vector3d> velocities = { Eigen::Vector3d(0, 0, 0) };
  vector<Eigen::Vector3d> velocities = { cur_vel };
  vector<double> yaws = { cur_yaw[0] };

  // Partitioning-based tour planning
  vector<int> ego_ids;
  vector<vector<int>> other_ids;
  if (!findGlobalTourOfGrid(positions, velocities, ego_ids, other_ids)) {
    grid_ids = {};
    {
      std::ostringstream oss;
      oss << "findGridAndFrontierPath no_grid pos=[" << cur_pos.transpose() << "]"
          << ", vel=[" << cur_vel.transpose() << "]";
      appendDiagLineFM(ep_->drone_id_, "TASK_ASSIGN", oss.str());
    }
    return;
  }
  grid_ids = ego_ids;

  double grid_time = (ros::Time::now() - t1).toSec();

  // Frontier-based single drone tour planning
  // Restrict frontier within the first visited grid
  t1 = ros::Time::now();

  vector<int> ftr_ids;
  // uniform_grid_->getFrontiersInGrid(ego_ids[0], ftr_ids);
  hgrid_->getFrontiersInGrid(ego_ids, ftr_ids);

  // Large maps and synchronized teammate maps can update/cover frontiers quickly.
  // hgrid may still hold stale frontier ids from the previous frontier set, so filter
  // them before constructing the TSP/cost matrix.
  const int frontier_num_now = frontier_finder_->getFrontierNum();
  vector<int> valid_ftr_ids;
  vector<int> invalid_ftr_ids;
  valid_ftr_ids.reserve(ftr_ids.size());
  for (const int fid : ftr_ids) {
    if (fid >= 0 && fid < frontier_num_now) valid_ftr_ids.push_back(fid);
    else invalid_ftr_ids.push_back(fid);
  }
  if (!invalid_ftr_ids.empty()) {
    ROS_WARN_THROTTLE(1.0,
        "[explore][drone %d] filtered %zu stale frontier ids, current_frontier_num=%d",
        ep_->drone_id_, invalid_ftr_ids.size(), frontier_num_now);
    std::ostringstream oss;
    oss << "filtered_stale_frontier_ids=" << vecToStrFM(invalid_ftr_ids)
        << ", current_frontier_num=" << frontier_num_now;
    appendDiagLineFM(ep_->drone_id_, "TASK_ASSIGN", oss.str());
  }
  ftr_ids.swap(valid_ftr_ids);

  const int max_ftr_for_tsp = 80;
  if ((int)ftr_ids.size() > max_ftr_for_tsp) {
    ROS_WARN_THROTTLE(1.0,
        "[explore][drone %d] too many frontier candidates %zu, truncate to %d",
        ep_->drone_id_, ftr_ids.size(), max_ftr_for_tsp);
    ftr_ids.resize(max_ftr_for_tsp);
  }

  ROS_INFO("Find frontier tour, %d involved------------", (int)ftr_ids.size());
  {
    std::ostringstream oss;
    oss << "findGridAndFrontierPath assigned_grid_ids=" << vecToStrFM(ego_ids)
        << ", other_grid_groups=" << vecVecToStrFM(other_ids)
        << ", frontier_candidates=" << vecToStrFM(ftr_ids)
        << ", frontier_candidate_count=" << ftr_ids.size()
        << ", current_frontier_num=" << frontier_num_now;
    appendDiagLineFM(ep_->drone_id_, "TASK_ASSIGN", oss.str());
  }

  if (ftr_ids.empty()) {
    frontier_ids = {};
    appendDiagLineFM(ep_->drone_id_, "TASK_ASSIGN", "findGridAndFrontierPath no_valid_frontier_in_assigned_grid");
    return;
  }

  // Consider next grid in frontier tour planning
  Eigen::Vector3d grid_pos;
  double grid_yaw;
  vector<Eigen::Vector3d> grid_pos_vec;
  if (hgrid_->getNextGrid(ego_ids, grid_pos, grid_yaw)) {
    grid_pos_vec = { grid_pos };
  }

  findTourOfFrontier(cur_pos, cur_vel, cur_yaw, ftr_ids, grid_pos_vec, frontier_ids);
  {
    std::ostringstream oss;
    oss << "findGridAndFrontierPath selected_frontier_ids=" << vecToStrFM(frontier_ids)
        << ", next_grid_points=" << grid_pos_vec.size();
    appendDiagLineFM(ep_->drone_id_, "TASK_ASSIGN", oss.str());
  }
  double ftr_time = (ros::Time::now() - t1).toSec();
  ROS_INFO("Grid tour t: %lf, frontier tour t: %lf.", grid_time, ftr_time);
}
//对路径进行简化
void FastExplorationManager::shortenPath(vector<Vector3d>& path) {
  if (path.empty()) {
    ROS_ERROR("Empty path to shorten");
    return;
  }
  // 缩短路线，仅保留关键中间点。
  const double dist_thresh = 3.0;
  vector<Vector3d> short_tour = { path.front() };
  for (int i = 1; i < path.size() - 1; ++i) {
    if ((path[i] - short_tour.back()).norm() > dist_thresh)
      short_tour.push_back(path[i]);//如果当前点与简化路径的最后一个点的距离大于 dist_thresh，保留当前点
    else {
      // 仅添加航点以缩短路径来避免碰撞
      ViewNode::caster_->input(short_tour.back(), path[i + 1]);
      Eigen::Vector3i idx;//如果当前点与简化路径的最后一个点的距离小于 dist_thresh，使用ViewNode::caster_检查碰撞
      while (ViewNode::caster_->nextId(idx) && ros::ok()) {
        if (edt_environment_->sdf_map_->getInflateOccupancy(idx) == 1 ||
            edt_environment_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
          short_tour.push_back(path[i]);
          break;
        }
      }
    }
  }
  if ((path.back() - short_tour.back()).norm() > 1e-3) short_tour.push_back(path.back());

  // 确保路径中至少包含三个点
  if (short_tour.size() == 2)
    short_tour.insert(short_tour.begin() + 1, 0.5 * (short_tour[0] + short_tour[1]));
  path = short_tour;
}
//直接对前沿点进行TSP
void FastExplorationManager::findGlobalTour(const Vector3d& cur_pos, const Vector3d& cur_vel,
    const Vector3d cur_yaw, vector<int>& indices) {
  auto t1 = ros::Time::now();

  // 获取当前状态与集群的成本矩阵
  Eigen::MatrixXd cost_mat;
  frontier_finder_->getFullCostMatrix(cur_pos, cur_vel, cur_yaw, cost_mat);
  const int dimension = cost_mat.rows();
  std::cout << "mat:   " << cost_mat.rows() << std::endl;

  double mat_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // 初始化TSP参数文件
  ofstream par_file(ep_->tsp_dir_ + "/drone_" + to_string(ep_->drone_id_) + ".par");
  par_file << "PROBLEM_FILE = " << ep_->tsp_dir_ + "/drone_" + to_string(ep_->drone_id_) + ".tsp\n";
  par_file << "GAIN23 = NO\n";
  par_file << "OUTPUT_TOUR_FILE ="
           << ep_->tsp_dir_ + "/drone_" + to_string(ep_->drone_id_) + ".tou"
                                                                      "r\n";
  par_file << "RUNS = 1\n";
  par_file.close();

  // 将参数和成本矩阵写入问题文件
  ofstream prob_file(ep_->tsp_dir_ + "/drone_" + to_string(ep_->drone_id_) + ".tsp");
  // 问题规范部分，遵循TSPLIB格式
  string prob_spec;
  prob_spec = "NAME : single\nTYPE : ATSP\nDIMENSION : " + to_string(dimension) +
              "\nEDGE_WEIGHT_TYPE : "
              "EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n";
  prob_file << prob_spec;
  // prob_file << "TYPE : TSP\n";
  // prob_file << "EDGE_WEIGHT_FORMAT : LOWER_ROW\n";
  // Problem data part
  const int scale = 100;
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = cost_mat(i, j) * scale;
      prob_file << int_cost << " ";
    }
    prob_file << "\n";
  }
  prob_file << "EOF";
  prob_file.close();

  // solveTSPLKH((ep_->tsp_dir_ + "/drone_" + to_string(ep_->drone_id_) + ".par").c_str());
  lkh_tsp_solver::SolveTSP srv;//调用TSP求解器
  if (!tsp_client_.call(srv)) {
    ROS_ERROR("Fail to solve TSP.");
    return;
  }

  // 从结果文件的游览部分读取最优路线
  ifstream res_file(ep_->tsp_dir_ + "/drone_" + to_string(ep_->drone_id_) + ".tour");
  string res;
  while (getline(res_file, res)) {
    // Go to tour section
    if (res.compare("TOUR_SECTION") == 0) break;
  }

  // 读取ATSP公式的路径
  while (getline(res_file, res)) {
    // 读取最优路线中的边界索引
    int id = stoi(res);
    if (id == 1)  // 忽略当前状态
      continue;
    if (id == -1) break;
    indices.push_back(id - 2);  // Idx of solver-2 == Idx of frontier
  }

  res_file.close();

  std::cout << "Tour " << ep_->drone_id_ << ": ";
  for (auto id : indices) std::cout << id << ", ";
  std::cout << "" << std::endl;

  // 从路径矩阵中获取最优游览的路径
  frontier_finder_->getPathForTour(cur_pos, indices, ed_->frontier_tour_);

  double tsp_time = (ros::Time::now() - t1).toSec();
  ROS_INFO("Cost mat: %lf, TSP: %lf", mat_time, tsp_time);

  // if (tsp_time > 0.1) ROS_BREAK();
}
//用 Dijkstra 搜索最优路径，提取位置和偏航角，再用A*提取局部路径用于可视化。
void FastExplorationManager::refineLocalTour(const Vector3d& cur_pos, const Vector3d& cur_vel,
    const Vector3d& cur_yaw, const vector<vector<Vector3d>>& n_points,
    const vector<vector<double>>& n_yaws, vector<Vector3d>& refined_pts,
    vector<double>& refined_yaws) {
  double create_time, search_time, parse_time;
  auto t1 = ros::Time::now();

  // Create graph for viewpoints selection
  GraphSearch<ViewNode> g_search;
  vector<ViewNode::Ptr> last_group, cur_group;

  // Add the current state
  ViewNode::Ptr first(new ViewNode(cur_pos, cur_yaw[0]));
  first->vel_ = cur_vel;
  g_search.addNode(first);
  last_group.push_back(first);
  ViewNode::Ptr final_node;

  // Add viewpoints
  std::cout << "Local refine graph size: 1, ";
  for (int i = 0; i < n_points.size(); ++i) {
    // Create nodes for viewpoints of one frontier
    for (int j = 0; j < n_points[i].size(); ++j) {
      ViewNode::Ptr node(new ViewNode(n_points[i][j], n_yaws[i][j]));
      g_search.addNode(node);
      // Connect a node to nodes in last group
      for (auto nd : last_group) g_search.addEdge(nd->id_, node->id_);
      cur_group.push_back(node);

      // Only keep the first viewpoint of the last local frontier
      if (i == n_points.size() - 1) {
        final_node = node;
        break;
      }
    }
    // Store nodes for this group for connecting edges
    std::cout << cur_group.size() << ", ";
    last_group = cur_group;
    cur_group.clear();
  }
  std::cout << "" << std::endl;
  create_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Search optimal sequence
  vector<ViewNode::Ptr> path;
  g_search.DijkstraSearch(first->id_, final_node->id_, path);

  search_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Return searched sequence
  for (int i = 1; i < path.size(); ++i) {
    refined_pts.push_back(path[i]->pos_);
    refined_yaws.push_back(path[i]->yaw_);
  }

  // Extract optimal local tour (for visualization)
  if (ep_->enable_refined_tour_vis_) {
    ed_->refined_tour_.clear();
    ed_->refined_tour_.push_back(cur_pos);
    ViewNode::astar_->lambda_heu_ = 1.0;
    ViewNode::astar_->setResolution(0.2);
    for (auto pt : refined_pts) {
      vector<Vector3d> path;
      if (ViewNode::searchPath(ed_->refined_tour_.back(), pt, path))
        ed_->refined_tour_.insert(ed_->refined_tour_.end(), path.begin(), path.end());
      else
        ed_->refined_tour_.push_back(pt);
    }
    ViewNode::astar_->lambda_heu_ = 10000;
  } else {
    ed_->refined_tour_.clear();
  }

  parse_time = (ros::Time::now() - t1).toSec();
  // ROS_WARN("create: %lf, search: %lf, parse: %lf", create_time, search_time, parse_time);
}
//网格分配
void FastExplorationManager::allocateGrids(const vector<Eigen::Vector3d>& positions,
    const vector<Eigen::Vector3d>& velocities, const vector<vector<int>>& first_ids,
    const vector<vector<int>>& second_ids, const vector<int>& grid_ids, vector<int>& ego_ids,
    vector<int>& other_ids) {
  // ROS_INFO("Allocate grid.");

  auto t1 = ros::Time::now();
  auto t2 = t1;

  if (grid_ids.size() == 1) {  // 只有一个网格，不需要运行CVRP
    auto pt = hgrid_->getCenter(grid_ids.front());//获取网格中心点
    // double d1 = (positions[0] - pt).norm();
    // double d2 = (positions[1] - pt).norm();
    vector<Eigen::Vector3d> path;//计算无人机到网格中心的成本
    double d1 = ViewNode::computeCost(positions[0], pt, 0, 0, Eigen::Vector3d(0, 0, 0), 0, path);
    double d2 = ViewNode::computeCost(positions[1], pt, 0, 0, Eigen::Vector3d(0, 0, 0), 0, path);
    if (d1 < d2) {
      ego_ids = grid_ids;
      other_ids = {};
    } else {
      ego_ids = {};
      other_ids = grid_ids;
    }
    return;
  }
  //计算无人机到网格的成本矩阵
  Eigen::MatrixXd mat;
  // uniform_grid_->getCostMatrix(positions, velocities, prev_first_ids, grid_ids, mat);
  hgrid_->getCostMatrix(positions, velocities, first_ids, second_ids, grid_ids, mat);
  applyRegionSelectionBias(positions, velocities, first_ids, grid_ids, mat);

  if (hgrid_->isTaskRegionMode()) {
    const int drone_num = positions.size();
    std::unordered_map<int, std::vector<int>> comp_map;
    for (int gid : grid_ids) comp_map[hgrid_->getRegionGroupId(gid)].push_back(gid);
    std::vector<std::vector<int>> comps;
    comps.reserve(comp_map.size());
    for (auto& kv : comp_map) comps.push_back(kv.second);

    auto comp_cost = [&](int di, const std::vector<int>& comp) {
      double best = 1e9;
      for (int gid : comp) {
        int gj = -1;
        for (int j = 0; j < (int)grid_ids.size(); ++j) if (grid_ids[j] == gid) { gj = j; break; }
        if (gj < 0) continue;
        best = std::min(best, mat(1 + di, 1 + drone_num + gj));
      }
      return best;
    };

    std::vector<std::vector<int>> assign(std::max(1, drone_num));
    std::vector<char> drone_taken(std::max(1, drone_num), 0), comp_taken(comps.size(), 0);
    struct Pair { double c; int i; int cg; };
    std::vector<Pair> pairs;
    for (int i = 0; i < drone_num; ++i)
      for (int cg = 0; cg < (int)comps.size(); ++cg)
        pairs.push_back({comp_cost(i, comps[cg]), i, cg});
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b){ return a.c < b.c; });
    for (const auto& p : pairs) {
      if (drone_taken[p.i] || comp_taken[p.cg]) continue;
      assign[p.i].insert(assign[p.i].end(), comps[p.cg].begin(), comps[p.cg].end());
      drone_taken[p.i] = 1; comp_taken[p.cg] = 1;
    }
    for (int cg = 0; cg < (int)comps.size(); ++cg) {
      if (comp_taken[cg]) continue;
      double best = 1e9; int best_i = 0;
      for (int i = 0; i < drone_num; ++i) {
        double c = comp_cost(i, comps[cg]);
        if (!assign[i].empty()) {
          double chain_bonus = 0.0;
          for (int gid : assign[i]) {
            for (int cg_gid : comps[cg]) {
              if (hgrid_->isConsistent(gid, cg_gid)) { chain_bonus -= 1.5; break; }
            }
          }
          c += chain_bonus;
        }
        if (c < best) { best = c; best_i = i; }
      }
      assign[best_i].insert(assign[best_i].end(), comps[cg].begin(), comps[cg].end());
    }
    for (int i = 0; i < drone_num; ++i) {
      std::sort(assign[i].begin(), assign[i].end());
      assign[i].erase(std::unique(assign[i].begin(), assign[i].end()), assign[i].end());
      reorderIdsGreedy(positions, velocities, first_ids, hgrid_.get(), assign[i]);
    }
    ego_ids = assign.empty() ? std::vector<int>() : assign[0];
    other_ids.clear();
    for (int i = 1; i < drone_num; ++i) {
      other_ids.insert(other_ids.end(), assign[i].begin(), assign[i].end());
    }
    return;
  }

  // int unknown = hgrid_->getTotalUnknwon();初始化CVRP参数
  int unknown;

  double mat_time = (ros::Time::now() - t1).toSec();

  // Find optimal path through AmTSP
  t1 = ros::Time::now();
  const int dimension = mat.rows();
  const int drone_num = positions.size();

  vector<int> unknown_nums;
  int capacity = 0;
  for (int i = 0; i < grid_ids.size(); ++i) {
    int unum = hgrid_->getUnknownCellsNum(grid_ids[i]);//获取每个网格的未知体素数
    unknown_nums.push_back(unum);
    capacity += unum;//总容量
    // std::cout << "Grid " << i << ": " << unum << std::endl;
  }
  // std::cout << "Total: " << capacity << std::endl;
  capacity = capacity * 0.75 * 0.1;

  // int prob_type;
  // if (grid_ids.size() >= 3)
  //   prob_type = 2;  // Use ACVRP
  // else
  //   prob_type = 1;  // Use AmTSP

  const int prob_type = 2;

  // Create problem file--------------------------
  ofstream file(ep_->mtsp_dir_ + "/amtsp3_" + to_string(ep_->drone_id_) + ".atsp");
  file << "NAME : pairopt\n";

  if (prob_type == 1)
    file << "TYPE : ATSP\n";
  else if (prob_type == 2)
    file << "TYPE : ACVRP\n";

  file << "DIMENSION : " + to_string(dimension) + "\n";
  file << "EDGE_WEIGHT_TYPE : EXPLICIT\n";
  file << "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n";

  if (prob_type == 2) {
    file << "CAPACITY : " + to_string(capacity) + "\n";   // ACVRP
    file << "VEHICLES : " + to_string(drone_num) + "\n";  // ACVRP
  }

  // Cost matrix
  file << "EDGE_WEIGHT_SECTION\n";
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = 100 * mat(i, j);
      file << int_cost << " ";
    }
    file << "\n";
  }

  if (prob_type == 2) {  // Demand section, ACVRP only
    file << "DEMAND_SECTION\n";
    file << "1 0\n";
    for (int i = 0; i < drone_num; ++i) {
      file << to_string(i + 2) + " 0\n";
    }
    for (int i = 0; i < grid_ids.size(); ++i) {
      int grid_unknown = unknown_nums[i] * 0.1;
      file << to_string(i + 2 + drone_num) + " " + to_string(grid_unknown) + "\n";
    }
    file << "DEPOT_SECTION\n";
    file << "1\n";
    file << "EOF";
  }

  file.close();

  // Create par file------------------------------------------
  int min_size = int(grid_ids.size()) / 2;
  int max_size = ceil(int(grid_ids.size()) / 2.0);
  file.open(ep_->mtsp_dir_ + "/amtsp3_" + to_string(ep_->drone_id_) + ".par");
  file << "SPECIAL\n";
  file << "PROBLEM_FILE = " + ep_->mtsp_dir_ + "/amtsp3_" + to_string(ep_->drone_id_) + ".atsp\n";
  if (prob_type == 1) {
    file << "SALESMEN = " << to_string(drone_num) << "\n";
    file << "MTSP_OBJECTIVE = MINSUM\n";
    // file << "MTSP_OBJECTIVE = MINMAX\n";
    file << "MTSP_MIN_SIZE = " << to_string(min_size) << "\n";
    file << "MTSP_MAX_SIZE = " << to_string(max_size) << "\n";
    file << "TRACE_LEVEL = 0\n";
  } else if (prob_type == 2) {
    file << "TRACE_LEVEL = 1\n";  // ACVRP
    file << "SEED = 0\n";         // ACVRP
  }
  file << "RUNS = 1\n";
  file << "TOUR_FILE = " + ep_->mtsp_dir_ + "/amtsp3_" + to_string(ep_->drone_id_) + ".tour\n";

  file.close();

  auto par_dir = ep_->mtsp_dir_ + "/amtsp3_" + to_string(ep_->drone_id_) + ".atsp";
  t1 = ros::Time::now();

  lkh_mtsp_solver::SolveMTSP srv;
  srv.request.prob = 3;
  // if (!tsp_client_.call(srv)) {
  if (!acvrp_client_.call(srv)) {
    ROS_ERROR("Fail to solve ACVRP.");
    return;
  }
  // system("/home/boboyu/software/LKH-3.0.6/LKH
  // /home/boboyu/workspaces/hkust_swarm_ws/src/swarm_exploration/utils/lkh_mtsp_solver/resource/amtsp3_1.par");

  double mtsp_time = (ros::Time::now() - t1).toSec();
  std::cout << "Allocation time: " << mtsp_time << std::endl;

  // Read results
  t1 = ros::Time::now();

  ifstream fin(ep_->mtsp_dir_ + "/amtsp3_" + to_string(ep_->drone_id_) + ".tour");
  string res;
  vector<int> ids;
  while (getline(fin, res)) {
    if (res.compare("TOUR_SECTION") == 0) break;
  }
  while (getline(fin, res)) {
    int id = stoi(res);
    ids.push_back(id - 1);
    if (id == -1) break;
  }
  fin.close();

  // Parse the m-tour of grid
  vector<vector<int>> tours;
  vector<int> tour;
  for (auto id : ids) {
    if (id > 0 && id <= drone_num) {
      tour.clear();
      tour.push_back(id);
    } else if (id >= dimension || id <= 0) {
      tours.push_back(tour);
    } else {
      tour.push_back(id);
    }
  }
  // // Print tour ids
  // for (auto tr : tours) {
  //   std::cout << "tour: ";
  //   for (auto id : tr) std::cout << id << ", ";
  //   std::cout << "" << std::endl;
  // }

  for (int i = 1; i < tours.size(); ++i) {
    if (tours[i][0] == 1) {
      ego_ids.insert(ego_ids.end(), tours[i].begin() + 1, tours[i].end());
    } else {
      other_ids.insert(other_ids.end(), tours[i].begin() + 1, tours[i].end());
    }
  }
  for (auto& id : ego_ids) {
    id = grid_ids[id - 1 - drone_num];
  }
  for (auto& id : other_ids) {
    id = grid_ids[id - 1 - drone_num];
  }
  // // Remove repeated grid
  // unordered_map<int, int> ego_map, other_map;
  // for (auto id : ego_ids) ego_map[id] = 1;
  // for (auto id : other_ids) other_map[id] = 1;

  // ego_ids.clear();
  // other_ids.clear();
  // for (auto p : ego_map) ego_ids.push_back(p.first);
  // for (auto p : other_map) other_ids.push_back(p.first);

  // sort(ego_ids.begin(), ego_ids.end());
  // sort(other_ids.begin(), other_ids.end());
}
double FastExplorationManager::computeRegionBiasForDrone(const int& drone_idx,
    const Eigen::Vector3d& pos, const Eigen::Vector3d& vel, const vector<int>& current_first_ids,
    const vector<int>& candidate_grid_ids, const int& grid_id, const int& available_region_num,
    const double mean_unknown_cells) const {
  double penalty = 0.0;
  const auto& states = ed_->swarm_state_;
  Eigen::Vector3d map_ori, map_size;
  sdf_map_->getRegion(map_ori, map_size);
  const AdaptivePenaltyContext ctx = buildAdaptivePenaltyContext(map_size, ep_->drone_num_);
  const double available_region_soft_min =
      std::max(ep_->dynamic_available_region_soft_min_, ctx.available_region_soft_min);
  const double avail_scale = clamp01(
      (double(available_region_num) - available_region_soft_min) /
      std::max(1.0, available_region_soft_min));

  const int cand_group = hgrid_->getRegionGroupId(grid_id);
  const double intent_vector_conflict_bias = computeIntentVectorConflictBias(
      drone_idx, pos, grid_id, candidate_grid_ids, states, hgrid_.get(), sdf_map_.get(),
      ctx, ep_->drone_num_);
  penalty += intent_vector_conflict_bias;

  Eigen::Vector2d cand_dir = (hgrid_->getCenter(grid_id) - pos).head<2>();
  double cand_norm = cand_dir.norm();
  if (cand_norm > 1e-3) cand_dir /= cand_norm;

  double ref_yaw = 0.0;
  bool have_ref_yaw = false;
  if (drone_idx >= 0 && drone_idx < (int)states.size() && states[drone_idx].stamp_ > 1e-3) {
    ref_yaw = states[drone_idx].yaw_;
    have_ref_yaw = true;
  } else if (vel.head<2>().norm() > 0.2) {
    ref_yaw = std::atan2(vel(1), vel(0));
    have_ref_yaw = true;
  }

  if (have_ref_yaw && cand_norm > 0.2) {
    const double cand_yaw = std::atan2(cand_dir(1), cand_dir(0));
    const double delta_yaw = wrapAnglePi(cand_yaw - ref_yaw);
    const double abs_delta_yaw = std::abs(delta_yaw);
    const bool have_sensor_fov = ep_->sensor_left_angle_ > 1e-2 && ep_->sensor_right_angle_ > 1e-2;
    const double fallback_hard_angle = std::acos(clampRange(ep_->dynamic_heading_hard_cos_, -1.0, 1.0));
    const double side_soft_angle = have_sensor_fov
        ? std::max(1e-2, delta_yaw >= 0.0 ? ep_->sensor_left_angle_ : ep_->sensor_right_angle_)
        : std::max(5.0 * M_PI / 180.0, 0.5 * fallback_hard_angle);
    const double combined_front_angle = have_sensor_fov
        ? std::max(side_soft_angle + 5.0 * M_PI / 180.0, ep_->sensor_left_angle_ + ep_->sensor_right_angle_)
        : fallback_hard_angle;
    const double hard_turn_angle = clampRange(combined_front_angle,
        side_soft_angle + 5.0 * M_PI / 180.0, M_PI - 5.0 * M_PI / 180.0);
    const double heading_penalty = std::max(0.1 * ep_->dynamic_heading_penalty_, ctx.heading_penalty);
    const double uturn_penalty = std::max(0.2 * ep_->dynamic_uturn_penalty_, ctx.uturn_penalty);
    const double soft_turn_gain = 0.45;
    const double hard_turn_gain = 0.55;

    if (abs_delta_yaw > side_soft_angle) {
      if (abs_delta_yaw <= hard_turn_angle) {
        const double t = (abs_delta_yaw - side_soft_angle) /
            std::max(1e-3, hard_turn_angle - side_soft_angle);
        penalty += soft_turn_gain * heading_penalty * t * t;
      } else {
        const double t = (abs_delta_yaw - hard_turn_angle) /
            std::max(1e-3, M_PI - hard_turn_angle);
        penalty += soft_turn_gain * heading_penalty + hard_turn_gain * uturn_penalty * t * t;
      }
    }
  }

  if (!current_first_ids.empty()) {
    int cur_gid = current_first_ids.front();
    int cur_group = hgrid_->getRegionGroupId(cur_gid);
    const double inertia_scale = 0.8 + ep_->task_inertia_available_scale_ * avail_scale;
    const double task_inertia_penalty = std::max(ep_->task_inertia_penalty_, 0.5 * ctx.task_inertia_penalty);
    const double task_keep_bonus = std::max(ep_->task_keep_bonus_, 0.5 * ctx.task_keep_bonus);
    if (grid_id == cur_gid || (cand_group >= 0 && cand_group == cur_group)) {
      penalty -= 1.8 * task_keep_bonus * inertia_scale;
    } else if (!hgrid_->isConsistent(cur_gid, grid_id)) {
      penalty += 1.8 * task_inertia_penalty * inertia_scale;
    } else {
      penalty += 0.8 * task_inertia_penalty * inertia_scale;
    }
  }

  penalty += 0.70 * computeSmallRegionCleanupBonus(
      hgrid_->getUnknownCellsNum(grid_id), hgrid_->getRegionTotalCellsNum(grid_id),
      mean_unknown_cells, available_region_num, ctx);
  penalty += computeNearbyIsolatedSmallRegionBonus(
      pos, grid_id, hgrid_.get(), ed_->swarm_state_[ep_->drone_id_ - 1].grid_ids_,
      mean_unknown_cells, available_region_num, ctx);
  return penalty;
}

void FastExplorationManager::applyRegionSelectionBias(const vector<Eigen::Vector3d>& positions,
    const vector<Eigen::Vector3d>& velocities, const vector<vector<int>>& first_ids,
    const vector<int>& grid_ids, Eigen::MatrixXd& mat) {
  if (grid_ids.empty()) return;
  const int drone_num = positions.size();
  const int grid_num = grid_ids.size();
  double mean_unknown_cells = 0.0;
  for (int gid : grid_ids) mean_unknown_cells += std::max(1, hgrid_->getUnknownCellsNum(gid));
  mean_unknown_cells /= std::max(1, grid_num);
  for (int i = 0; i < drone_num; ++i) {
    const vector<int> cur_first = (i < (int)first_ids.size()) ? first_ids[i] : vector<int>{};
    for (int j = 0; j < grid_num; ++j) {
      double bias = computeRegionBiasForDrone(i, positions[i], velocities[i], cur_first,
          grid_ids, grid_ids[j], grid_num, mean_unknown_cells);
      mat(1 + i, 1 + drone_num + j) += bias;
    }
  }
}

//网格访问路径成本计算
double FastExplorationManager::computeGridPathCost(const Eigen::Vector3d& pos,
    const vector<int>& grid_ids, const vector<int>& first, const vector<vector<int>>& firsts,
    const vector<vector<int>>& seconds, const double& w_f) {
  if (grid_ids.empty()) return 0.0;

  double cost = 0.0;
  vector<Eigen::Vector3d> path;
  cost += hgrid_->getCostDroneToGrid(pos, grid_ids[0], first);
  for (int i = 0; i < grid_ids.size() - 1; ++i) {
    cost += hgrid_->getCostGridToGrid(grid_ids[i], grid_ids[i + 1], firsts, seconds, firsts.size());
  }
  return cost;
}
//全局路径规划
bool FastExplorationManager::findGlobalTourOfGrid(const vector<Eigen::Vector3d>& positions,
    const vector<Eigen::Vector3d>& velocities, vector<int>& indices, vector<vector<int>>& others,
    bool init) {

  ROS_INFO("Find grid tour---------------");

  auto t1 = ros::Time::now();

  auto& grid_ids = ed_->swarm_state_[ep_->drone_id_ - 1].grid_ids_;//获取当前无人机的网格ID列表

  // hgrid_->updateBaseCoor();  // Use the latest basecoor transform of swarm

  vector<int> first_ids, second_ids;
  hgrid_->inputFrontiers(ed_->averages_);//将前沿点输入到网格管理器中

  hgrid_->updateGridData(
      ep_->drone_id_, grid_ids, ed_->reallocated_, ed_->last_grid_ids_, first_ids, second_ids);//更新网格数据
  {
    std::ostringstream oss;
    oss << "findGlobalTourOfGrid raw_candidates=" << vecToStrFM(grid_ids)
        << ", first_ids=" << vecToStrFM(first_ids)
        << ", second_ids=" << vecToStrFM(second_ids)
        << ", last_grid_ids=" << vecToStrFM(ed_->last_grid_ids_)
        << ", reallocated=" << (ed_->reallocated_ ? "true" : "false")
        << ", init=" << (init ? "true" : "false");
    appendDiagLineFM(ep_->drone_id_, "TASK_ASSIGN", oss.str());
  }

  Eigen::Vector3d map_ori, map_size;
  sdf_map_->getRegion(map_ori, map_size);
  const AdaptivePenaltyContext adaptive_ctx = buildAdaptivePenaltyContext(map_size, ep_->drone_num_);

  if (hgrid_->isTaskRegionMode()) {
    first_ids = inferSpatialContinuationIds(positions[0], velocities[0], hgrid_.get(), grid_ids, adaptive_ctx);
    second_ids.clear();
  }

  if (grid_ids.empty()) {
    ROS_WARN("Empty dominance.");
    appendDiagLineFM(ep_->drone_id_, "TASK_ASSIGN", "findGlobalTourOfGrid empty_dominance_after_updateGridData");
    ed_->grid_tour_.clear();
    return false;
  }

  if (hgrid_->isTaskRegionMode() && !grid_ids.empty()) {
    std::vector<std::pair<double,int>> scored;
    double mean_unknown_cells = 0.0;
    for (int gid : grid_ids) mean_unknown_cells += std::max(1, hgrid_->getUnknownCellsNum(gid));
    mean_unknown_cells /= std::max<size_t>(1, grid_ids.size());
    for (int gid : grid_ids) {
      double c = hgrid_->getCostDroneToGrid(positions[0], gid, first_ids);
      c += computeRegionBiasForDrone(ep_->drone_id_ - 1, positions[0], velocities[0], first_ids,
          grid_ids, gid, (int)grid_ids.size(), mean_unknown_cells);
      c += 0.35 * computeSmallRegionCleanupBonus(
          hgrid_->getUnknownCellsNum(gid), hgrid_->getRegionTotalCellsNum(gid),
          mean_unknown_cells, (int)grid_ids.size(), adaptive_ctx);
      c += 0.85 * computeNearbyIsolatedSmallRegionBonus(
          positions[0], gid, hgrid_.get(), grid_ids, mean_unknown_cells, (int)grid_ids.size(), adaptive_ctx);
      scored.push_back({c, gid});
    }
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
    int keep_n = std::min((int)scored.size(), std::max(5, std::min(10, ep_->greedy_tsp_max_nodes_)));
    std::vector<int> shortlisted; shortlisted.reserve(keep_n);
    std::unordered_set<int> chosen_groups;
    for (auto& sg : scored) {
      int g = hgrid_->getRegionGroupId(sg.second);
      if (chosen_groups.insert(g).second) shortlisted.push_back(sg.second);
      if ((int)shortlisted.size() >= keep_n) break;
    }
    for (auto& sg : scored) {
      if ((int)shortlisted.size() >= keep_n) break;
      if (std::find(shortlisted.begin(), shortlisted.end(), sg.second) == shortlisted.end()) shortlisted.push_back(sg.second);
    }
    grid_ids.swap(shortlisted);
    {
      std::ostringstream oss;
      oss << "findGlobalTourOfGrid shortlisted_candidates=" << vecToStrFM(grid_ids)
          << ", scored_count=" << scored.size();
      appendDiagLineFM(ep_->drone_id_, "TASK_ASSIGN", oss.str());
    }
  }

  std::cout << "Allocated grid: ";
  for (auto id : grid_ids) std::cout << id << ", ";
  std::cout << "" << std::endl;

  Eigen::MatrixXd mat;//计算成本矩阵
  // uniform_grid_->getCostMatrix(positions, velocities, first_ids, grid_ids, mat);
  if (!init)
    hgrid_->getCostMatrix(positions, velocities, { first_ids }, { second_ids }, grid_ids, mat);
  else
    hgrid_->getCostMatrix(positions, velocities, { {} }, { {} }, grid_ids, mat);
  applyRegionSelectionBias(positions, velocities, { first_ids }, grid_ids, mat);

  double mat_time = (ros::Time::now() - t1).toSec();

  // Fast greedy fallback for task-region mode or small problems to avoid repeated solver stalls
  if ((hgrid_->isTaskRegionMode() && ep_->enable_fast_greedy_tsp_) ||
      (ep_->enable_fast_greedy_tsp_ && (int)grid_ids.size() <= ep_->greedy_tsp_max_nodes_)) {
    auto order_nodes = greedyOrderFromMatrix(mat, 1);
    indices.clear();
    for (auto node_id : order_nodes) {
      int local_id = node_id - 2;  // 1 depot + 1 drone
      if (local_id >= 0 && local_id < (int)grid_ids.size()) indices.push_back(grid_ids[local_id]);
    }
    others.clear();
    grid_ids = indices;
    hgrid_->getGridTour(grid_ids, positions[0], ed_->grid_tour_, ed_->grid_tour2_);
    ed_->last_grid_ids_ = grid_ids;
    ed_->reallocated_ = false;
    {
      std::ostringstream oss;
      oss << "findGlobalTourOfGrid greedy_selected_indices=" << vecToStrFM(indices)
          << ", selected_grid_ids=" << vecToStrFM(grid_ids)
          << ", grid_tour_size=" << ed_->grid_tour_.size();
      appendDiagLineFM(ep_->drone_id_, "TASK_ASSIGN", oss.str());
    }
    return true;
  }

  // Find optimal path through ATSP
  t1 = ros::Time::now();
  const int dimension = mat.rows();
  const int drone_num = 1;//独立规划自己的网格

  // Create problem file 创建atsp文件（问题描述）
  ofstream file(ep_->mtsp_dir_ + "/amtsp2_" + to_string(ep_->drone_id_) + ".atsp");
  file << "NAME : amtsp\n";
  file << "TYPE : ATSP\n";
  file << "DIMENSION : " + to_string(dimension) + "\n";
  file << "EDGE_WEIGHT_TYPE : EXPLICIT\n";
  file << "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n";
  file << "EDGE_WEIGHT_SECTION\n";
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = 100 * mat(i, j);
      file << int_cost << " ";
    }
    file << "\n";
  }
  file.close();

  // Create par file  创建LKH参数文件
  file.open(ep_->mtsp_dir_ + "/amtsp2_" + to_string(ep_->drone_id_) + ".par");
  file << "SPECIAL\n";
  file << "PROBLEM_FILE = " + ep_->mtsp_dir_ + "/amtsp2_" + to_string(ep_->drone_id_) + ".atsp\n";
  file << "SALESMEN = " << to_string(drone_num) << "\n";
  file << "MTSP_OBJECTIVE = MINSUM\n";
  // file << "MTSP_MIN_SIZE = " << to_string(min(int(ed_->frontiers_.size()) / drone_num, 4)) <<
  // "\n"; file << "MTSP_MAX_SIZE = "
  //      << to_string(max(1, int(ed_->frontiers_.size()) / max(1, drone_num - 1))) << "\n";
  file << "RUNS = 1\n";
  file << "TRACE_LEVEL = 0\n";
  file << "TOUR_FILE = " + ep_->mtsp_dir_ + "/amtsp2_" + to_string(ep_->drone_id_) + ".tour\n";
  file.close();

  auto par_dir = ep_->mtsp_dir_ + "/amtsp2_" + to_string(ep_->drone_id_) + ".atsp";
  t1 = ros::Time::now();

  lkh_mtsp_solver::SolveMTSP srv;
  srv.request.prob = 2;
  if (!tsp_client_.call(srv)) {
    ROS_ERROR("Fail to solve ATSP.");
    return false;
  }

  double mtsp_time = (ros::Time::now() - t1).toSec();
  // std::cout << "AmTSP time: " << mtsp_time << std::endl;

  // Read results
  t1 = ros::Time::now();

  ifstream fin(ep_->mtsp_dir_ + "/amtsp2_" + to_string(ep_->drone_id_) + ".tour");
  string res;
  vector<int> ids;
  while (getline(fin, res)) {
    if (res.compare("TOUR_SECTION") == 0) break;
  }
  while (getline(fin, res)) {
    int id = stoi(res);
    ids.push_back(id - 1);
    if (id == -1) break;
  }
  fin.close();

  // Parse the m-tour of grid
  vector<vector<int>> tours;
  vector<int> tour;
  for (auto id : ids) {
    if (id > 0 && id <= drone_num) {
      tour.clear();
      tour.push_back(id);
    } else if (id >= dimension || id <= 0) {
      tours.push_back(tour);
    } else {
      tour.push_back(id);
    }
  }

  // for (auto tr : tours) {
  //   std::cout << "tour: ";
  //   for (auto id : tr) std::cout << id << ", ";
  //   std::cout << "" << std::endl;
  // }
  others.resize(drone_num - 1);
  for (int i = 1; i < tours.size(); ++i) {
    if (tours[i][0] == 1) {
      indices.insert(indices.end(), tours[i].begin() + 1, tours[i].end());
    } else {
      others[tours[i][0] - 2].insert(
          others[tours[i][0] - 2].end(), tours[i].begin(), tours[i].end());
    }
  }
  for (auto& id : indices) {
    id -= 1 + drone_num;
  }
  for (auto& other : others) {
    for (auto& id : other) id -= 1 + drone_num;
  }
  std::cout << "Grid tour: ";
  for (auto& id : indices) {
    id = grid_ids[id];
    std::cout << id << ", ";
  }
  std::cout << "" << std::endl;

  // uniform_grid_->getGridTour(indices, ed_->grid_tour_);
  grid_ids = indices;
  hgrid_->getGridTour(grid_ids, positions[0], ed_->grid_tour_, ed_->grid_tour2_);

  ed_->last_grid_ids_ = grid_ids;
  ed_->reallocated_ = false;
  {
    std::ostringstream oss;
    oss << "findGlobalTourOfGrid mtsp_selected_indices=" << vecToStrFM(indices)
        << ", selected_grid_ids=" << vecToStrFM(grid_ids)
        << ", others=" << vecVecToStrFM(others)
        << ", grid_tour_size=" << ed_->grid_tour_.size();
    appendDiagLineFM(ep_->drone_id_, "TASK_ASSIGN", oss.str());
  }

  // hgrid_->checkFirstGrid(grid_ids.front());

  return true;
}
//为无人机在分配的网格内寻找前沿点并规划路径
void FastExplorationManager::findTourOfFrontier(const Vector3d& cur_pos, const Vector3d& cur_vel,
    const Vector3d& cur_yaw, const vector<int>& ftr_ids, const vector<Eigen::Vector3d>& grid_pos,
    vector<int>& indices) {

  auto t1 = ros::Time::now();

  vector<Eigen::Vector3d> positions = { cur_pos };
  vector<Eigen::Vector3d> velocities = { cur_vel };
  vector<double> yaws = { cur_yaw[0] };

  // frontier_finder_->getSwarmCostMatrix(positions, velocities, yaws, mat);
  Eigen::MatrixXd mat;
  frontier_finder_->getSwarmCostMatrix(positions, velocities, yaws, ftr_ids, grid_pos, mat);
  if (mat.rows() <= 0 || mat.cols() <= 0) {
    ROS_ERROR("[explore][drone %d] empty frontier TSP matrix, skip frontier tour", ep_->drone_id_);
    indices.clear();
    return;
  }
  const int dimension = mat.rows();
  // std::cout << "dim of frontier TSP mat: " << dimension << std::endl;

  double mat_time = (ros::Time::now() - t1).toSec();
  // ROS_INFO("mat time: %lf", mat_time);

  if ((hgrid_->isTaskRegionMode() && ep_->enable_fast_greedy_tsp_) ||
      (ep_->enable_fast_greedy_tsp_ && (int)ftr_ids.size() <= ep_->greedy_tsp_max_nodes_)) {
    auto order_nodes = greedyOrderFromMatrix(mat, 1);
    indices.clear();
    for (auto node_id : order_nodes) {
      int local_id = node_id - 2;
      if (local_id >= 0 && local_id < (int)ftr_ids.size()) indices.push_back(ftr_ids[local_id]);
    }
    frontier_finder_->getPathForTour(cur_pos, indices, ed_->frontier_tour_);
    if (!grid_pos.empty()) ed_->frontier_tour_.push_back(grid_pos[0]);
    return;
  }

  // Find optimal allocation through AmTSP
  t1 = ros::Time::now();

  // Create problem file
  ofstream file(ep_->mtsp_dir_ + "/amtsp_" + to_string(ep_->drone_id_) + ".atsp");
  file << "NAME : amtsp\n";
  file << "TYPE : ATSP\n";
  file << "DIMENSION : " + to_string(dimension) + "\n";
  file << "EDGE_WEIGHT_TYPE : EXPLICIT\n";
  file << "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n";
  file << "EDGE_WEIGHT_SECTION\n";
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = 100 * mat(i, j);
      file << int_cost << " ";
    }
    file << "\n";
  }
  file.close();

  // Create par file
  const int drone_num = 1;

  file.open(ep_->mtsp_dir_ + "/amtsp_" + to_string(ep_->drone_id_) + ".par");
  file << "SPECIAL\n";
  file << "PROBLEM_FILE = " + ep_->mtsp_dir_ + "/amtsp_" + to_string(ep_->drone_id_) + ".atsp\n";
  file << "SALESMEN = " << to_string(drone_num) << "\n";
  file << "MTSP_OBJECTIVE = MINSUM\n";
  file << "MTSP_MIN_SIZE = " << to_string(min(int(ed_->frontiers_.size()) / drone_num, 4)) << "\n";
  file << "MTSP_MAX_SIZE = "
       << to_string(max(1, int(ed_->frontiers_.size()) / max(1, drone_num - 1))) << "\n";
  file << "RUNS = 1\n";
  file << "TRACE_LEVEL = 0\n";
  file << "TOUR_FILE = " + ep_->mtsp_dir_ + "/amtsp_" + to_string(ep_->drone_id_) + ".tour\n";
  file.close();

  auto par_dir = ep_->mtsp_dir_ + "/amtsp_" + to_string(ep_->drone_id_) + ".atsp";
  t1 = ros::Time::now();

  lkh_mtsp_solver::SolveMTSP srv;
  srv.request.prob = 1;
  if (!tsp_client_.call(srv)) {
    ROS_ERROR("Fail to solve ATSP.");
    return;
  }

  double mtsp_time = (ros::Time::now() - t1).toSec();
  // ROS_INFO("AmTSP time: %lf", mtsp_time);

  // Read results
  t1 = ros::Time::now();

  ifstream fin(ep_->mtsp_dir_ + "/amtsp_" + to_string(ep_->drone_id_) + ".tour");
  string res;
  vector<int> ids;
  while (getline(fin, res)) {
    if (res.compare("TOUR_SECTION") == 0) break;
  }
  while (getline(fin, res)) {
    int id = stoi(res);
    ids.push_back(id - 1);
    if (id == -1) break;
  }
  fin.close();

  // Parse the m-tour
  vector<vector<int>> tours;
  vector<int> tour;
  for (auto id : ids) {
    if (id > 0 && id <= drone_num) {
      tour.clear();
      tour.push_back(id);
    } else if (id >= dimension || id <= 0) {
      tours.push_back(tour);
    } else {
      tour.push_back(id);
    }
  }

  vector<vector<int>> others(drone_num - 1);
  for (int i = 1; i < tours.size(); ++i) {
    if (tours[i][0] == 1) {
      indices.insert(indices.end(), tours[i].begin() + 1, tours[i].end());
    }
    // else {
    //   others[tours[i][0] - 2].insert(
    //       others[tours[i][0] - 2].end(), tours[i].begin() + 1, tours[i].end());
    // }
  }
  for (auto& id : indices) {
    id -= 1 + drone_num;
  }
  // for (auto& other : others) {
  //   for (auto& id : other)
  //     id -= 1 + drone_num;
  // }

  if (ed_->grid_tour_.size() > 2) {  // Remove id for next grid, since it is considered in the TSP
    indices.pop_back();
  }
  // Subset of frontier inside first grid
  for (int i = 0; i < indices.size(); ++i) {
    indices[i] = ftr_ids[indices[i]];
  }

  // Get the path of optimal tour from path matrix
  frontier_finder_->getPathForTour(cur_pos, indices, ed_->frontier_tour_);
  if (!grid_pos.empty()) {
    ed_->frontier_tour_.push_back(grid_pos[0]);
  }

  // ed_->other_tours_.clear();
  // for (int i = 1; i < positions.size(); ++i) {
  //   ed_->other_tours_.push_back({});
  //   frontier_finder_->getPathForTour(positions[i], others[i - 1], ed_->other_tours_[i - 1]);
  // }

  double parse_time = (ros::Time::now() - t1).toSec();
  // ROS_INFO("Cost mat: %lf, TSP: %lf, parse: %f, %d frontiers assigned.", mat_time, mtsp_time,
  //     parse_time, indices.size());
}

}  // namespace fast_planner
