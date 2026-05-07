#include <deque>
#include <Eigen/Eigenvalues>
#include <active_perception/uniform_grid.h>
#include <active_perception/hgrid.h>
#include <active_perception/graph_node.h>
#include <path_searching/astar2.h>
#include <plan_env/sdf_map.h>
#include <plan_env/edt_environment.h>

namespace fast_planner {

HGrid::HGrid(const shared_ptr<EDTEnvironment>& edt, ros::NodeHandle& nh) {

  this->edt_ = edt;
  nh.param("partitioning/consistent_cost", consistent_cost_, 3.5);
  nh.param("partitioning/consistent_cost2", consistent_cost2_, 3.5);
  nh.param("partitioning/use_swarm_tf", use_swarm_tf_, false);
  nh.param("partitioning/w_first", w_first_, 1.0);
  nh.param("partitioning/use_task_region_decision", use_task_region_decision_, true);
  nh.param("partitioning/task_cluster_radius", task_cluster_radius_, 2.4);
  nh.param("partitioning/task_band_radius", task_band_radius_, 1.5);
  nh.param("partitioning/task_region_inflate_xy", task_region_inflate_xy_, 0.5);
  nh.param("partitioning/task_z_tol", task_z_tol_, 0.6);
  nh.param("partitioning/task_axis_neighbor_radius", task_axis_neighbor_radius_, 3.5);
  nh.param("partitioning/task_half_span", task_half_span_, 2.5);
  nh.param("partitioning/task_seed_radius", task_seed_radius_, 1.0);
  nh.param("partitioning/task_overlap_bias", task_overlap_bias_, 1.5);
  nh.param("partitioning/task_rebuild_min_interval", task_rebuild_min_interval_, 0.25);
  nh.param("partitioning/task_rebuild_frontier_shift", task_rebuild_frontier_shift_, 0.35);
  nh.param("partitioning/task_anchor_resolution", task_anchor_resolution_, 1.0);
  nh.param("partitioning/task_min_frontiers", task_min_frontiers_, 1);
  nh.param("partitioning/task_max_frontiers_per_region", task_max_frontiers_per_region_, 5);
  nh.param("partitioning/task_max_anchor_regions", task_max_anchor_regions_, 18);
  nh.param("partitioning/task_min_free_cells", task_min_free_cells_, 3);
  nh.param("partitioning/task_min_unknown_cells", task_min_unknown_cells_, 3);
  last_task_region_build_time_ = ros::Time(0);

  path_finder_.reset(new Astar);
  path_finder_->init(nh, edt);

  grid1_.reset(new UniformGrid(edt, nh, 1));
  grid2_.reset(new UniformGrid(edt, nh, 2));

  // Swarm tf
  grid1_->use_swarm_tf_ = grid2_->use_swarm_tf_ = use_swarm_tf_;
  double yaw = 0.0;
  rot_sw_ << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1;
  trans_sw_ << 0.0, 0.0, 0;
  grid1_->rot_sw_ = grid2_->rot_sw_ = rot_sw_;
  grid1_->trans_sw_ = grid2_->trans_sw_ = trans_sw_;

  // Wait for swarm basecoor transform and initialize grid
  // while (!updateBaseCoor()) {
  //   ROS_WARN("Wait for basecoor.");
  //   ros::Duration(0.5).sleep();
  //   ros::spinOnce();
  // }
  grid1_->initGridData();
  grid2_->initGridData();
  updateBaseCoor();
}

HGrid::~HGrid() {
}

bool HGrid::updateBaseCoor() {

  // Eigen::Vector4d tf;
  // if (!edt_->sdf_map_->getBaseCoor(1, tf)) return false;
  // double yaw = tf[3];
  // rot_sw_ << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1;
  // trans_sw_ = tf.head<3>();

  rot_sw_ = Eigen::Matrix3d::Identity();
  trans_sw_ = Eigen::Vector3d::Zero();

  grid1_->rot_sw_ = grid2_->rot_sw_ = rot_sw_;
  grid1_->trans_sw_ = grid2_->trans_sw_ = trans_sw_;
  grid1_->updateBaseCoor();
  grid2_->updateBaseCoor();

  return true;
}

void HGrid::inputFrontiers(const vector<Eigen::Vector3d>& avgs) {
  frontier_avgs_ = avgs;
  // Keep original grids updated for compatibility and visualization fallback.
  grid1_->inputFrontiers(avgs);
  grid2_->inputFrontiers(avgs);
  if (use_task_region_decision_ && shouldRebuildTaskRegions(avgs)) {
    buildTaskRegions();
    last_task_region_build_time_ = ros::Time::now();
    last_frontier_count_ = avgs.size();
    if (!avgs.empty()) {
      last_frontier_centroid_.setZero();
      for (const auto& p : avgs) last_frontier_centroid_ += p;
      last_frontier_centroid_ /= double(avgs.size());
    }
  }
}

bool HGrid::shouldRebuildTaskRegions(const vector<Eigen::Vector3d>& avgs) const {
  if (last_task_region_build_time_.isZero()) return true;
  if (avgs.empty() != (last_frontier_count_ <= 0)) return true;
  if (std::abs((int)avgs.size() - last_frontier_count_) >= 2) return true;
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  if (!avgs.empty()) {
    for (const auto& p : avgs) centroid += p;
    centroid /= double(avgs.size());
  }
  double shift = (centroid - last_frontier_centroid_).norm();
  double dt = (ros::Time::now() - last_task_region_build_time_).toSec();
  return shift >= task_rebuild_frontier_shift_ || dt >= task_rebuild_min_interval_;
}

void HGrid::buildTaskRegions() {

  task_regions_.clear();
  task_region_cells_.clear();
  task_region_addr_sets_.clear();
  task_region_anchors_.clear();
  task_region_tangents_.clear();
  task_region_group_ids_.clear();
  if (!use_task_region_decision_ || frontier_avgs_.empty()) return;

  auto map = edt_->sdf_map_;
  const double res = map->getResolution();

  auto addr_of = [&](const Eigen::Vector3i& idx) { return map->toAddress(idx); };
  auto in_safe_box = [&](const Eigen::Vector3i& idx) {
    return map->isInMap(idx) && map->isInBox(idx);
  };
  auto is_free = [&](const Eigen::Vector3i& idx) {
    return in_safe_box(idx) && map->getInflateOccupancy(idx) != 1 &&
           map->getOccupancy(idx) == SDFMap::FREE;
  };
  auto has_mixed_contact = [&](const Eigen::Vector3i& idx, bool& adj_free, bool& adj_unknown) {
    adj_free = false;
    adj_unknown = false;
    static const int nbrs[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    for (int k = 0; k < 6; ++k) {
      Eigen::Vector3i nid = idx + Eigen::Vector3i(nbrs[k][0], nbrs[k][1], nbrs[k][2]);
      if (!in_safe_box(nid)) continue;
      int occ = map->getOccupancy(nid);
      if (occ == SDFMap::FREE && map->getInflateOccupancy(nid) != 1) adj_free = true;
      if (occ == SDFMap::UNKNOWN) adj_unknown = true;
    }
  };
  auto estimate_tangent_from_ids = [&](const std::vector<int>& ids) {
    if (ids.empty()) return Eigen::Vector2d(1.0, 0.0);
    std::vector<Eigen::Vector2d> samples;
    samples.reserve(ids.size());
    for (int id : ids) samples.push_back(frontier_avgs_[id].head<2>());
    if (samples.size() >= 2) {
      Eigen::Vector2d mean = Eigen::Vector2d::Zero();
      for (auto& p : samples) mean += p;
      mean /= double(samples.size());
      Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
      for (auto& p : samples) {
        Eigen::Vector2d d = p - mean;
        cov += d * d.transpose();
      }
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
      Eigen::Vector2d axis = es.eigenvectors().col(1);
      if (axis.norm() > 1e-3) return axis.normalized();
    }
    return Eigen::Vector2d(1.0, 0.0);
  };

  struct SegmentDef {
    std::vector<int> fids;
    Eigen::Vector2d anchor;
    Eigen::Vector2d tangent;
  };
  std::vector<SegmentDef> segments;

  // Cluster frontiers first to avoid fragmented noisy regions in dense areas.
  std::vector<char> visited(frontier_avgs_.size(), 0);
  for (int i = 0; i < (int)frontier_avgs_.size(); ++i) {
    if (visited[i]) continue;
    std::vector<int> cluster;
    std::deque<int> q;
    q.push_back(i);
    visited[i] = 1;
    while (!q.empty()) {
      int u = q.front(); q.pop_front();
      cluster.push_back(u);
      for (int v = 0; v < (int)frontier_avgs_.size(); ++v) {
        if (visited[v]) continue;
        if (std::abs(frontier_avgs_[v].z() - frontier_avgs_[u].z()) > task_z_tol_) continue;
        if ((frontier_avgs_[v].head<2>() - frontier_avgs_[u].head<2>()).norm() > task_cluster_radius_) continue;
        visited[v] = 1;
        q.push_back(v);
      }
    }
    if ((int)cluster.size() < task_min_frontiers_) continue;

    Eigen::Vector2d tangent = estimate_tangent_from_ids(cluster);
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (int id : cluster) mean += frontier_avgs_[id].head<2>();
    mean /= double(cluster.size());

    std::vector<std::pair<double, int>> proj;
    proj.reserve(cluster.size());
    for (int id : cluster) {
      double s = (frontier_avgs_[id].head<2>() - mean).dot(tangent);
      proj.emplace_back(s, id);
    }
    std::sort(proj.begin(), proj.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) { return a.first < b.first; });

    int max_seg_frontiers = std::max(2, task_max_frontiers_per_region_);
    int st = 0;
    while (st < (int)proj.size()) {
      int ed = st;
      while (ed + 1 < (int)proj.size()) {
        double gap = proj[ed + 1].first - proj[ed].first;
        double span = proj[ed + 1].first - proj[st].first;
        int next_cnt = ed + 1 - st + 1;
        if (gap > std::max(task_band_radius_, 1.2 * res) ||
            span > 2.2 * task_half_span_ ||
            next_cnt > max_seg_frontiers) {
          break;
        }
        ++ed;
      }
      std::vector<int> seg_ids;
      for (int k = st; k <= ed; ++k) seg_ids.push_back(proj[k].second);
      if ((int)seg_ids.size() >= task_min_frontiers_) {
        Eigen::Vector2d anchor = Eigen::Vector2d::Zero();
        for (int id : seg_ids) anchor += frontier_avgs_[id].head<2>();
        anchor /= double(seg_ids.size());
        segments.push_back({seg_ids, anchor, estimate_tangent_from_ids(seg_ids)});
      }
      st = ed + 1;
    }
  }

  if (segments.empty()) return;
  if ((int)segments.size() > task_max_anchor_regions_) {
    std::sort(segments.begin(), segments.end(), [](const SegmentDef& a, const SegmentDef& b) {
      return a.fids.size() > b.fids.size();
    });
    segments.resize(task_max_anchor_regions_);
  }

  struct Candidate {
    GridInfo info;
    std::vector<Eigen::Vector3i> cells;
    std::unordered_set<int> addrs;
    Eigen::Vector2d anchor;
    Eigen::Vector2d tangent;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(segments.size());

  for (const auto& seg : segments) {
    Eigen::Vector2d tangent = seg.tangent;
    if (tangent.norm() < 1e-3) tangent = Eigen::Vector2d(1.0, 0.0);
    tangent.normalize();
    Eigen::Vector2d normal(-tangent.y(), tangent.x());
    normal.normalize();

    double z_mean = 0.0;
    for (int id : seg.fids) z_mean += frontier_avgs_[id].z();
    z_mean /= double(seg.fids.size());
    double half_span = task_half_span_;
    for (int id : seg.fids) {
      Eigen::Vector2d d = frontier_avgs_[id].head<2>() - seg.anchor;
      half_span = std::max(half_span, std::fabs(d.dot(tangent)) + 0.6 * res);
    }

    Eigen::Vector3d low(seg.anchor.x() - std::max(half_span, task_band_radius_) - task_region_inflate_xy_,
                        seg.anchor.y() - std::max(half_span, task_band_radius_) - task_region_inflate_xy_,
                        z_mean - task_z_tol_ - res);
    Eigen::Vector3d up(seg.anchor.x() + std::max(half_span, task_band_radius_) + task_region_inflate_xy_,
                       seg.anchor.y() + std::max(half_span, task_band_radius_) + task_region_inflate_xy_,
                       z_mean + task_z_tol_ + res);
    map->boundBox(low, up);
    Eigen::Vector3i id_low, id_up;
    map->posToIndex(low, id_low);
    map->posToIndex(up, id_up);
    map->boundIndex(id_low);
    map->boundIndex(id_up);

    std::unordered_set<int> admissible;
    std::vector<Eigen::Vector3i> free_seeds, unknown_seeds;
    for (int x = id_low.x(); x <= id_up.x(); ++x) {
      for (int y = id_low.y(); y <= id_up.y(); ++y) {
        for (int z = id_low.z(); z <= id_up.z(); ++z) {
          Eigen::Vector3i idx(x, y, z);
          if (!in_safe_box(idx)) continue;
          Eigen::Vector3d pos; map->indexToPos(idx, pos);
          if (std::fabs(pos.z() - z_mean) > task_z_tol_ + 0.51 * res) continue;
          Eigen::Vector2d dxy = pos.head<2>() - seg.anchor;
          if (std::fabs(dxy.dot(tangent)) > half_span + 0.51 * res) continue;
          if (std::fabs(dxy.dot(normal)) > task_band_radius_ + 0.51 * res) continue;
          int occ = map->getOccupancy(idx);
          if (occ != SDFMap::FREE && occ != SDFMap::UNKNOWN) continue;
          if (occ == SDFMap::FREE && map->getInflateOccupancy(idx) == 1) continue;
          admissible.insert(addr_of(idx));
          bool adj_free = false, adj_unknown = false;
          has_mixed_contact(idx, adj_free, adj_unknown);
          bool near_frontier = false;
          for (int fid : seg.fids) {
            if ((pos.head<2>() - frontier_avgs_[fid].head<2>()).norm() <= task_seed_radius_ + 0.75 * res) {
              near_frontier = true;
              break;
            }
          }
          if (near_frontier) {
            if (occ == SDFMap::FREE && adj_unknown) free_seeds.push_back(idx);
            if (occ == SDFMap::UNKNOWN && adj_free) unknown_seeds.push_back(idx);
          }
        }
      }
    }

    if (free_seeds.empty() || unknown_seeds.empty()) continue;

    std::unordered_set<int> visited_cells;
    std::deque<Eigen::Vector3i> q;
    auto push = [&](const Eigen::Vector3i& idx) {
      int addr = addr_of(idx);
      if (admissible.find(addr) == admissible.end()) return;
      if (visited_cells.insert(addr).second) q.push_back(idx);
    };
    for (const auto& s : free_seeds) push(s);
    for (const auto& s : unknown_seeds) push(s);

    static const int nbrs[10][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{1,1,0},{1,-1,0},{-1,1,0},{-1,-1,0},{0,0,1},{0,0,-1}};
    std::vector<Eigen::Vector3i> cells;
    while (!q.empty()) {
      auto cur = q.front(); q.pop_front();
      cells.push_back(cur);
      for (auto& d : nbrs) {
        Eigen::Vector3i nxt = cur + Eigen::Vector3i(d[0], d[1], d[2]);
        if (!in_safe_box(nxt)) continue;
        int addr = addr_of(nxt);
        if (admissible.find(addr) == admissible.end()) continue;
        if (visited_cells.insert(addr).second) q.push_back(nxt);
      }
    }

    if (cells.empty()) continue;
    int free_cnt = 0, unknown_cnt = 0;
    for (auto& idx : cells) {
      int occ = map->getOccupancy(idx);
      if (occ == SDFMap::FREE) ++free_cnt;
      else if (occ == SDFMap::UNKNOWN) ++unknown_cnt;
    }
    if (free_cnt < task_min_free_cells_ || unknown_cnt < task_min_unknown_cells_) continue;

    Candidate cand;
    cand.anchor = seg.anchor;
    cand.tangent = tangent;
    cand.info.active_ = true;
    cand.info.is_cur_relevant_ = true;
    cand.info.is_prev_relevant_ = true;
    cand.info.need_divide_ = false;
    cand.info.frontier_num_ = seg.fids.size();
    cand.info.unknown_num_ = unknown_cnt;
    for (int fid : seg.fids) {
      cand.info.contained_frontier_ids_[fid] = 1;
      cand.info.frontier_cell_nums_[fid] = 1;
    }
    cand.cells = std::move(cells);
    for (const auto& idx : cand.cells) cand.addrs.insert(addr_of(idx));
    candidates.push_back(std::move(cand));
  }

  if (candidates.empty()) return;

  struct OwnerInfo { int rid; double score; Eigen::Vector3i idx; };
  std::unordered_map<int, OwnerInfo> owner;
  for (int rid = 0; rid < (int)candidates.size(); ++rid) {
    Eigen::Vector2d normal(-candidates[rid].tangent.y(), candidates[rid].tangent.x());
    normal.normalize();
    for (auto& idx : candidates[rid].cells) {
      int addr = addr_of(idx);
      Eigen::Vector3d pos; map->indexToPos(idx, pos);
      Eigen::Vector2d dxy = pos.head<2>() - candidates[rid].anchor;
      double score = std::fabs(dxy.dot(candidates[rid].tangent)) +
                     task_overlap_bias_ * std::fabs(dxy.dot(normal));
      auto it = owner.find(addr);
      if (it == owner.end() || score < it->second.score) owner[addr] = {rid, score, idx};
    }
  }

  std::vector<std::vector<Eigen::Vector3i>> owned_cells(candidates.size());
  std::vector<std::unordered_set<int>> owned_sets(candidates.size());
  for (auto& kv : owner) {
    int rid = kv.second.rid;
    owned_cells[rid].push_back(kv.second.idx);
    owned_sets[rid].insert(kv.first);
  }

  auto largest_component = [&](const std::vector<Eigen::Vector3i>& cells,
                               const std::unordered_set<int>& addrset) {
    std::vector<Eigen::Vector3i> best;
    std::unordered_set<int> seen;
    static const int nbrs[8][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{1,1,0},{1,-1,0},{-1,1,0},{-1,-1,0}};
    for (const auto& s : cells) {
      int sadd = addr_of(s);
      if (seen.count(sadd)) continue;
      std::deque<Eigen::Vector3i> q;
      std::vector<Eigen::Vector3i> comp;
      q.push_back(s); seen.insert(sadd);
      while (!q.empty()) {
        auto cur = q.front(); q.pop_front();
        comp.push_back(cur);
        for (auto& d : nbrs) {
          Eigen::Vector3i nxt = cur + Eigen::Vector3i(d[0], d[1], d[2]);
          int addr = addr_of(nxt);
          if (addrset.find(addr) == addrset.end() || seen.count(addr)) continue;
          seen.insert(addr); q.push_back(nxt);
        }
      }
      if (comp.size() > best.size()) best.swap(comp);
    }
    return best;
  };

  for (int rid = 0; rid < (int)candidates.size(); ++rid) {
    auto cells = largest_component(owned_cells[rid], owned_sets[rid]);
    if (cells.empty()) continue;
    std::unordered_set<int> addrset;
    for (const auto& idx : cells) addrset.insert(addr_of(idx));

    int free_cnt = 0, unknown_cnt = 0;
    Eigen::Vector3d free_sum = Eigen::Vector3d::Zero(), unknown_sum = Eigen::Vector3d::Zero();
    for (auto& idx : cells) {
      int occ = map->getOccupancy(idx);
      Eigen::Vector3d pos; map->indexToPos(idx, pos);
      if (occ == SDFMap::FREE) { ++free_cnt; free_sum += pos; }
      else if (occ == SDFMap::UNKNOWN) { ++unknown_cnt; unknown_sum += pos; }
    }
    if (free_cnt < task_min_free_cells_ || unknown_cnt < task_min_unknown_cells_) continue;
    GridInfo region = candidates[rid].info;
    region.unknown_num_ = unknown_cnt;
    Eigen::Vector3d free_center = free_sum / std::max(1, free_cnt);
    Eigen::Vector3d unknown_center = unknown_sum / std::max(1, unknown_cnt);
    Eigen::Vector3d desired_center = 0.5 * (free_center + unknown_center);
    double best = 1e9;
    Eigen::Vector3d center = free_center;
    for (auto& idx : cells) {
      if (!is_free(idx)) continue;
      Eigen::Vector3d pos; map->indexToPos(idx, pos);
      double d = (pos - desired_center).norm();
      if (d < best) { best = d; center = pos; }
    }
    region.center_ = center;
    task_regions_.push_back(region);
    task_region_cells_.push_back(cells);
    task_region_addr_sets_.push_back(addrset);
    task_region_anchors_.push_back(candidates[rid].anchor);
    task_region_tangents_.push_back(candidates[rid].tangent.normalized());
    task_region_group_ids_.push_back((int)task_region_group_ids_.size());
  }

  if (task_regions_.size() <= 1) return;

  // Build region groups: strongly overlapping / front-back consecutive regions should be treated as one task family.
  const int n = task_regions_.size();
  std::vector<int> parent(n);
  for (int i = 0; i < n; ++i) parent[i] = i;
  auto findp = [&](auto&& self, int x) -> int { return parent[x] == x ? x : parent[x] = self(self, parent[x]); };
  auto unite = [&](int a, int b) { a = findp(findp, a); b = findp(findp, b); if (a != b) parent[b] = a; };
  auto frontier_gap = [&](int i, int j) {
    double best = 1e9;
    for (const auto& fi : task_regions_[i].contained_frontier_ids_) {
      for (const auto& fj : task_regions_[j].contained_frontier_ids_) {
        best = std::min(best, (frontier_avgs_[fi.first].head<2>() - frontier_avgs_[fj.first].head<2>()).norm());
      }
    }
    return best;
  };
  auto shared_frontier_ratio = [&](int i, int j) {
    int inter = 0;
    for (const auto& fi : task_regions_[i].contained_frontier_ids_) if (task_regions_[j].contained_frontier_ids_.count(fi.first)) ++inter;
    int denom = std::max(1, std::min((int)task_regions_[i].contained_frontier_ids_.size(), (int)task_regions_[j].contained_frontier_ids_.size()));
    return double(inter) / double(denom);
  };
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      Eigen::Vector2d ti = task_region_tangents_[i];
      Eigen::Vector2d tj = task_region_tangents_[j];
      if (ti.norm() < 1e-3 || tj.norm() < 1e-3) continue;
      ti.normalize(); tj.normalize();
      double align = std::abs(ti.dot(tj));
      Eigen::Vector2d avg_t = (ti + tj);
      if (avg_t.norm() < 1e-3) avg_t = ti;
      avg_t.normalize();
      Eigen::Vector2d avg_n(-avg_t.y(), avg_t.x());
      Eigen::Vector2d d = task_regions_[j].center_.head<2>() - task_regions_[i].center_.head<2>();
      double long_gap = std::abs(d.dot(avg_t));
      double norm_gap = std::abs(d.dot(avg_n));
      double fgap = frontier_gap(i, j);
      double shared = shared_frontier_ratio(i, j);
      bool overlap_like = align > 0.78 && norm_gap < 0.9 * task_band_radius_ &&
                          long_gap < 1.0 * task_half_span_ &&
                          (shared > 0.15 || fgap < 0.9 * task_axis_neighbor_radius_);
      bool chain_like = align > 0.82 && norm_gap < 1.15 * task_band_radius_ &&
                        long_gap < 2.4 * task_half_span_ &&
                        fgap < 1.25 * task_axis_neighbor_radius_;
      if (overlap_like || chain_like) unite(i, j);
    }
  }
  std::unordered_map<int,int> remap;
  int gid = 0;
  for (int i = 0; i < n; ++i) {
    int r = findp(findp, i);
    if (!remap.count(r)) remap[r] = gid++;
    task_region_group_ids_[i] = remap[r];
  }
}

void HGrid::updateGridData(const int& drone_id, vector<int>& grid_ids, bool reallocated,
    const vector<int>& last_grid_ids, vector<int>& first_ids, vector<int>& second_ids) {

  if (use_task_region_decision_) {
    // Task-region mode should not inherit the previous queue as the current candidate set.
    // Otherwise newly spawned continuation regions around current position (e.g. R5 after R1)
    // never enter the candidate pool, and the planner is forced to keep following the stale queue.
    first_ids.clear();
    second_ids.clear();
    grid_ids.clear();
    for (int i = 0; i < (int)task_regions_.size(); ++i) grid_ids.push_back(i);
    return;
  }

  // Convert grid_ids to the ids of bi-level uniform grid
  vector<int> grid_ids1, grid_ids2;
  const int grid_num1 = grid1_->grid_data_.size();
  for (auto id : grid_ids) {
    if (id < grid_num1)
      grid_ids1.push_back(id);
    else
      grid_ids2.push_back(id - grid_num1);  // Id of level 2 grid
  }

  // std::cout << "Input ids: ";
  // for (auto id : grid_ids)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // std::cout << "level 1 ids: ";
  // for (auto id : grid_ids1)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // std::cout << "level 2 ids: ";
  // for (auto id : grid_ids2)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // Update at level 1
  vector<int> tmp_ids1 = grid_ids1;
  vector<int> parti_ids1, parti_ids1_all;
  grid1_->updateGridData(drone_id, grid_ids1, parti_ids1, parti_ids1_all);

  // std::cout << "updated level 1 ids: ";
  // for (auto id : grid_ids1)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // std::cout << "divided level 1 ids: ";
  // for (auto id : parti_ids1)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // Merge the newly partitioned and original grid ids
  vector<int> fine_ids;
  for (auto id : parti_ids1) {
    vector<int> tmp_ids;
    coarseToFineId(id, tmp_ids);
    fine_ids.insert(fine_ids.end(), tmp_ids.begin(), tmp_ids.end());
  }
  grid_ids2.insert(grid_ids2.end(), fine_ids.begin(), fine_ids.end());

  // Activate newly divided grids
  vector<int> fine_ids_all;
  for (auto id : parti_ids1_all) {
    vector<int> tmp_ids;
    coarseToFineId(id, tmp_ids);
    fine_ids_all.insert(fine_ids_all.end(), tmp_ids.begin(), tmp_ids.end());
  }
  grid2_->activateGrids(fine_ids_all);

  // if (reallocated) grid2_->activateGrids(grid_ids2);

  // std::cout << "merged level 2 ids: ";
  // for (auto id : grid_ids2)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  vector<int> parti_ids2, parti_ids2_all;  // Should be empty, no partition at level 2
  grid2_->updateGridData(drone_id, grid_ids2, parti_ids2, parti_ids2_all);

  // std::cout << "updated level 2 ids: ";
  // for (auto id : grid_ids2)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  grid_ids = grid_ids1;
  for (auto& id : grid_ids2) {
    grid_ids.push_back(id + grid_num1);
  }

  // Maintain consistency of next visited grid
  // if (reallocated) return;
  getConsistentGrid(last_grid_ids, grid_ids, first_ids, second_ids);
}

void HGrid::getConsistentGrid(const vector<int>& last_ids, const vector<int>& cur_ids,
    vector<int>& first_ids, vector<int>& second_ids) {

  if (last_ids.empty()) return;
  // std::cout << "last id: ";
  // for (auto id : last_ids)
  //   std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // Find the first two level 1 grids in last sequence
  const int grid_num1 = grid1_->grid_data_.size();
  int grid_id1 = last_ids[0];
  if (grid_id1 >= grid_num1) {
    int tmp = grid_id1 - grid_num1;
    fineToCoarseId(tmp, grid_id1);
  }

  // std::cout << "level 1 grid 1: " << grid_id1 << std::endl;

  int grid_id2 = -1;
  for (int i = 1; i < last_ids.size(); ++i) {
    if (last_ids[i] < grid_num1) {
      grid_id2 = last_ids[i];
      break;
    } else {
      int fine = last_ids[i] - grid_num1;
      int coarse;
      fineToCoarseId(fine, coarse);
      if (coarse != grid_id1) {
        grid_id2 = coarse;
        break;
      }
    }
  }
  // std::cout << "level 1 grid 2: " << grid_id2 << std::endl;

  first_ids.clear();
  // In the current sequence, try to find the first level 1 grid...
  for (auto id : cur_ids) {
    if (id == grid_id1) {
      first_ids = { id };
      // std::cout << "1" << std::endl;
    }
  }

  if (first_ids.empty()) {
    // or its sub-grids
    for (auto id : cur_ids) {
      if (id < grid_num1) continue;
      int coarse;
      fineToCoarseId(id - grid_num1, coarse);
      if (coarse == grid_id1) {
        first_ids.push_back(id);
      }
    }
  }

  vector<int>* ids_ptr;
  if (!first_ids.empty()) {
    // Already find the first, should find the second
    ids_ptr = &second_ids;
  } else {
    // No first yet, continue to find the first
    ids_ptr = &first_ids;
  }

  // Can not find first grid/sub-grid, try to find the second one
  if (grid_id2 == -1) return;

  for (auto id : cur_ids) {
    if (id == grid_id2) {
      *ids_ptr = { id };
      // std::cout << "3" << std::endl;
      return;
    }
  }

  for (auto id : cur_ids) {
    if (id < grid_num1) continue;
    int coarse;
    fineToCoarseId(id - grid_num1, coarse);
    if (coarse == grid_id2) {
      // std::cout << "4" << std::endl;
      ids_ptr->push_back(id);
    }
  }
  return;
}

void HGrid::coarseToFineId(const int& coarse, vector<int>& fines) {
  // 0: 0, 1
  // 1: 2, 3
  // 2: 4, 5
  fines.clear();
  Eigen::Vector3i cidx;  // coarse idx
  grid1_->adrToIndex(coarse, cidx);

  vector<Eigen::Vector3i> fine_idxs;
  fine_idxs.emplace_back(cidx[0] * 2, cidx[1] * 2, cidx[2]);
  fine_idxs.emplace_back(cidx[0] * 2 + 1, cidx[1] * 2, cidx[2]);
  fine_idxs.emplace_back(cidx[0] * 2, cidx[1] * 2 + 1, cidx[2]);
  fine_idxs.emplace_back(cidx[0] * 2 + 1, cidx[1] * 2 + 1, cidx[2]);

  for (auto idx : fine_idxs) {
    fines.push_back(grid2_->toAddress(idx));
  }
}
//将第二层级（细粒度）的网格ID转换为第一层级（粗粒度）的网格ID
void HGrid::fineToCoarseId(const int& fine, int& coarse) {
  Eigen::Vector3i fidx;//三维索引，表示第二层级网格在空间中的位置
  grid2_->adrToIndex(fine, fidx);//将第二层级的网格ID fine 转换为三维索引 fidx

  Eigen::Vector3i cidx;//三维索引，表示第一层级网格在空间中的位置
  cidx[0] = fidx[0] / 2;
  cidx[1] = fidx[1] / 2;
  cidx[2] = fidx[2];

  coarse = grid1_->toAddress(cidx);//将第一层级的索引转换为id
}
//获取成本矩阵
void HGrid::getCostMatrix(const vector<Eigen::Vector3d>& positions,
    const vector<Eigen::Vector3d>& velocities, const vector<vector<int>>& first_ids,
    const vector<vector<int>>& second_ids, const vector<int>& grid_ids, Eigen::MatrixXd& mat) {
  // first_ids and second_ids are drone_num x 1-4 vectors
  if (use_task_region_decision_) {
    const int drone_num = positions.size();
    const int reg_num = grid_ids.size();
    const int dimen = 1 + drone_num + reg_num;
    mat = Eigen::MatrixXd::Constant(dimen, dimen, 1000.0);
    for (int i = 0; i < drone_num; ++i) { mat(0,1+i) = -1000; mat(1+i,0)=1000; }
    for (int i = 0; i < reg_num; ++i) { mat(0,1+drone_num+i)=1000; mat(1+drone_num+i,0)=0; }
    for (int i = 0; i < drone_num; ++i) for (int j = 0; j < drone_num; ++j) mat(1+i,1+j)=10000;
    for (int i = 0; i < drone_num; ++i) for (int j = 0; j < reg_num; ++j) {
      double c = getCostDroneToGrid(positions[i], grid_ids[j], first_ids[i]);
      mat(1+i,1+drone_num+j)=c; mat(1+drone_num+j,1+i)=0;
    }
    for (int i = 0; i < reg_num; ++i) for (int j = i+1; j < reg_num; ++j) {
      double c = getCostGridToGrid(grid_ids[i], grid_ids[j], first_ids, second_ids, drone_num);
      mat(1+drone_num+i,1+drone_num+j)=c; mat(1+drone_num+j,1+drone_num+i)=c;
    }
    for (int i=0;i<dimen;++i) mat(i,i)=1000;
    return;
  }

  // Fill the cost matrix
  const int drone_num = positions.size();//无人机数量
  const int grid_num = grid_ids.size();//网格数量
  const int dimen = 1 + drone_num + grid_num;//成本矩阵维度
  mat = Eigen::MatrixXd::Zero(dimen, dimen);//初始化为零矩阵

  // std::cout << "First id: ";
  // for (auto ids : first_ids)
  //   for (auto id : ids)
  //     std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // std::cout << "Second id: ";
  // for (auto ids : second_ids)
  //   for (auto id : ids)
  //     std::cout << id << ", ";
  // std::cout << "" << std::endl;

  // Virtual depot to drones
  for (int i = 0; i < drone_num; ++i) {//虚拟仓库到无人机的成本
    mat(0, 1 + i) = -1000;//虚拟仓库到无人机成本为-1000，表示无人机从虚拟仓库出发
    mat(1 + i, 0) = 1000;//无人机到虚拟仓库的成本设置为 1000，表示无人机返回虚拟仓库
  }
  // Virtual depot to grid
  for (int i = 0; i < grid_num; ++i) {
    mat(0, 1 + drone_num + i) = 1000;//虚拟仓库到网格的成本设置为 1000
    mat(1 + drone_num + i, 0) = 0;//网格到虚拟仓库的成本设置为 0
  }
  // Costs between drones
  for (int i = 0; i < drone_num; ++i) {//无人机之间的成本
    for (int j = 0; j < drone_num; ++j) {
      mat(1 + i, 1 + j) = 10000;//表示无人机之间不能转移
    }
  }

  // Costs from drones to grid
  for (int i = 0; i < drone_num; ++i) {//无人机到网格的成本
    for (int j = 0; j < grid_num; ++j) {
      double cost = getCostDroneToGrid(positions[i], grid_ids[j], first_ids[i]) ;
                    - 5*getMinDist2DToBoundary1(positions[i]) - 5*getMinDist2DToBoundary2(positions[i]);//自己加的
      mat(1 + i, 1 + drone_num + j) = cost;
      mat(1 + drone_num + j, 1 + i) = 0;//表示网格不能直接返回无人机
    }
  }
  // Costs between grid
  for (int i = 0; i < grid_num; ++i) {
    for (int j = i + 1; j < grid_num; ++j) {//计算网格之间的成本
      double cost = getCostGridToGrid(grid_ids[i], grid_ids[j], first_ids, second_ids, drone_num);
      mat(1 + drone_num + i, 1 + drone_num + j) = cost;
      mat(1 + drone_num + j, 1 + drone_num + i) = cost;
    }
  }

  // Diag
  for (int i = 0; i < dimen; ++i) {
    mat(i, i) = 1000;//对角线成本，表示节点到自身的成本
  }
}
//无人机到网格之间的成本计算
double HGrid::getCostDroneToGrid(
    const Eigen::Vector3d& pos, const int& grid_id, const vector<int>& first) {
  auto& grid = getGrid(grid_id);
  double dist1 = (pos - grid.center_).norm();
  double cost;
  if (dist1 < 5.0) {
    path_finder_->reset();
    if (path_finder_->search(pos, grid.center_) == Astar::REACH_END) {
      auto path = path_finder_->getPath();
      cost = path_finder_->pathLength(path);
    } else {
      cost = dist1 + consistent_cost2_;
    }
  } else {
    //cost = 1.5 * dist1 + consistent_cost2_;
    cost = dist1 + consistent_cost2_;
  }
  if (!first.empty()) {
    for (auto first_id : first) {
      if (grid_id == first_id) {
        if (use_task_region_decision_)
          cost -= consistent_cost_;
        else
          cost += consistent_cost_;
        break;
      }
    }
  }
  return cost;
}

double HGrid::getCostGridToGrid(const int& id1, const int& id2, const vector<vector<int>>& firsts,
    const vector<vector<int>>& seconds, const int& drone_num) {
  auto& grid1 = getGrid(id1);
  auto& grid2 = getGrid(id2);
  double dist_cost = 0.0;
  if (isClose(id1, id2)) {
    path_finder_->reset();
    if (path_finder_->search(grid1.center_, grid2.center_) == Astar::REACH_END) {
      auto path = path_finder_->getPath();
      dist_cost = path_finder_->pathLength(path);
    } else {
      dist_cost = (grid1.center_ - grid2.center_).norm() + consistent_cost2_;
    }
    if (!use_task_region_decision_ && drone_num <= 1 && inSameLevel1(id1, id2)) {
      dist_cost += consistent_cost_;
    }
    if (!firsts.empty()) {
      bool is_first = false, is_second = false;
      for (int k = 0; k < drone_num; ++k) {
        is_first = false; is_second = false;
        for (auto first_id : firsts[k]) if (id1 == first_id) { is_first = true; break; }
        for (auto second_id : seconds[k]) if (id2 == second_id) { is_second = true; break; }
        if (is_first && is_second) break;
      }
      if (is_first && is_second) dist_cost += consistent_cost_;
    }
  } else {
    dist_cost = 1.5 * (grid1.center_ - grid2.center_).norm() + consistent_cost2_;
  }
  return dist_cost;
}

//获取指定网格中的未知单元格数量
//获取指定网格中的未知单元格数量
int HGrid::getUnknownCellsNum(const int& grid_id) {
  if (use_task_region_decision_) {
    if (grid_id < 0 || grid_id >= (int)task_regions_.size()) return 0;
    return task_regions_[grid_id].unknown_num_;
  }
  return getGrid(grid_id).unknown_num_;
}

int HGrid::getRegionTotalCellsNum(const int& grid_id) {
  if (use_task_region_decision_) {
    if (grid_id < 0 || grid_id >= (int)task_region_cells_.size()) return 0;
    return (int)task_region_cells_[grid_id].size();
  }
  auto& grid = getGrid(grid_id);
  return std::max(0, grid.unknown_num_ + grid.frontier_num_);
}

double HGrid::getRegionUnknownRatio(const int& grid_id) {
  const int total_cells = getRegionTotalCellsNum(grid_id);
  if (total_cells <= 0) return 1.0;
  return std::max(0.0, std::min(1.0, double(getUnknownCellsNum(grid_id)) / double(total_cells)));
}

Eigen::Vector3d HGrid::getCenter(const int& id) {
  if (use_task_region_decision_) {
    if (id < 0 || id >= (int)task_regions_.size()) return Eigen::Vector3d::Zero();
    return task_regions_[id].center_;
  }
  return getGrid(id).center_;
}

int HGrid::getRegionGroupId(const int& id) const {
  if (!use_task_region_decision_) return id;
  if (id < 0 || id >= (int)task_region_group_ids_.size()) return -1;
  return task_region_group_ids_[id];
}

GridInfo& HGrid::getGrid(const int& id) {
  if (use_task_region_decision_) {
    return task_regions_[id];
  }
  int grid_num1 = grid1_->grid_data_.size();
  if (id < grid_num1)
    return grid1_->grid_data_[id];
  else
    return grid2_->grid_data_[id - grid_num1];
}

void HGrid::getActiveGrids(vector<int>& grid_ids) {
  grid_ids.clear();
  if (use_task_region_decision_) {
    for (int i = 0; i < (int)task_regions_.size(); ++i) grid_ids.push_back(i);
    return;
  }
  const int grid_num1 = grid1_->grid_data_.size();
  for (int i = 0; i < grid_num1; ++i) {
    if (grid1_->grid_data_[i].active_ && grid1_->grid_data_[i].is_cur_relevant_) grid_ids.push_back(i);
  }
  for (int i = 0; i < (int)grid2_->grid_data_.size(); ++i) {
    if (grid2_->grid_data_[i].active_ && grid2_->grid_data_[i].is_cur_relevant_) grid_ids.push_back(i + grid_num1);
  }
}

bool HGrid::getNextGrid(const vector<int>& grid_ids, Eigen::Vector3d& grid_pos, double& grid_yaw) {
  if (use_task_region_decision_) {
    if (grid_ids.size() < 2) return false;
    auto& g1 = task_regions_[grid_ids[0]];
    auto& g2 = task_regions_[grid_ids[1]];
    grid_pos = g2.center_;
    Eigen::Vector3d dir = g2.center_ - g1.center_;
    grid_yaw = atan2(dir[1], dir[0]);
    return true;
  }
  const int grid_num1 = grid1_->grid_data_.size();
  int grid_id1;
  if (grid_ids[0] < grid_num1) grid_id1 = grid_ids[0];
  else { int fine = grid_ids[0] - grid_num1; fineToCoarseId(fine, grid_id1); }
  int grid_id2 = -1;
  for (int i = 1; i < (int)grid_ids.size(); ++i) {
    if (grid_ids[i] < grid_num1) {
      grid_id2 = grid_ids[i];
      break;
    }
    int fine = grid_ids[i] - grid_num1; int coarse; fineToCoarseId(fine, coarse);
    if (coarse != grid_id1) {
      grid_id2 = grid_ids[i];
      break;
    }
  }
  if (grid_id2 == -1) return false;
  auto& grid1 = getGrid(grid_id1);
  auto& grid2 = getGrid(grid_id2);
  grid_pos = grid2.center_;
  Eigen::Vector3d dir = grid2.center_ - grid1.center_;
  grid_yaw = atan2(dir[1], dir[0]);
  return true;
}

bool HGrid::isClose(const int& id1, const int& id2) {
  if (use_task_region_decision_) {
    if (id1 < 0 || id2 < 0 || id1 >= (int)task_regions_.size() || id2 >= (int)task_regions_.size()) return false;
    return (task_regions_[id1].center_ - task_regions_[id2].center_).norm() <= 2.0 * task_half_span_ + 2.0 * task_band_radius_;
  }
  const int grid_num1 = grid1_->grid_data_.size();
  int tmp_id1 = id1;
  if (tmp_id1 >= grid_num1) {
    int fine = tmp_id1 - grid_num1;
    fineToCoarseId(fine, tmp_id1);
  }
  int tmp_id2 = id2;
  if (tmp_id2 >= grid_num1) {
    int fine = tmp_id2 - grid_num1;
    fineToCoarseId(fine, tmp_id2);
  }
  Eigen::Vector3i idx1, idx2;
  grid1_->adrToIndex(tmp_id1, idx1);
  grid1_->adrToIndex(tmp_id2, idx2);
  for (int i = 0; i < 3; ++i) {
    if (abs(idx1[i] - idx2[i]) > 1) return false;
  }
  return true;
}

bool HGrid::inSameLevel1(const int& id1, const int& id2) {
  const int grid_num1 = grid1_->grid_data_.size();
  if (id1 < grid_num1 || id2 < grid_num1) return false;
  int tmp1 = id1 - grid_num1;
  int tmp2 = id2 - grid_num1;
  int coarse1, coarse2;
  fineToCoarseId(tmp1, coarse1);
  fineToCoarseId(tmp2, coarse2);
  return coarse1 == coarse2;
}

bool HGrid::isConsistent(const int& id1, const int& id2) {
  if (use_task_region_decision_) {
    if (id1 < 0 || id2 < 0 || id1 >= (int)task_regions_.size() || id2 >= (int)task_regions_.size()) return false;
    if (id1 < (int)task_region_group_ids_.size() && id2 < (int)task_region_group_ids_.size() &&
        task_region_group_ids_[id1] >= 0 && task_region_group_ids_[id1] == task_region_group_ids_[id2]) return true;
    return (task_regions_[id1].center_.head<2>() - task_regions_[id2].center_.head<2>()).norm() <= (task_axis_neighbor_radius_ + 0.8 * task_half_span_);
  }
  const int grid_num1 = grid1_->grid_data_.size();
  int tmp1 = id1;
  if (tmp1 >= grid_num1) { tmp1 -= grid_num1; int coarse; fineToCoarseId(tmp1, coarse); tmp1 = coarse; }
  int tmp2 = id2;
  if (tmp2 >= grid_num1) { tmp2 -= grid_num1; int coarse; fineToCoarseId(tmp2, coarse); tmp2 = coarse; }
  return tmp1 == tmp2;
}

void HGrid::getGridTour(const vector<int>& ids, const Eigen::Vector3d& pos,
    vector<Eigen::Vector3d>& tour, vector<Eigen::Vector3d>& tour2) {
  if (use_task_region_decision_) {
    vector<Eigen::Vector3d> centers = { pos };
    for (int id : ids) if (id >= 0 && id < (int)task_regions_.size()) centers.push_back(task_regions_[id].center_);
    tour = centers;
    tour2 = { pos };
    for (int i = 0; i < (int)centers.size() - 1; ++i) {
      path_finder_->reset();
      if (path_finder_->search(centers[i], centers[i + 1]) == Astar::REACH_END) {
        auto path = path_finder_->getPath();
        tour2.insert(tour2.end(), path.begin() + 1, path.end());
      } else {
        tour2.push_back(centers[i + 1]);
      }
    }
    return;
  }
  const int grid_num1 = grid1_->grid_data_.size();
  vector<Eigen::Vector3d> centers = { pos };
  for (auto id : ids) {
    if (id < grid_num1) centers.push_back(grid1_->grid_data_[id].center_);
    else centers.push_back(grid2_->grid_data_[id - grid_num1].center_);
  }
  tour = centers;
  tour2 = { pos };
  for (int i = 0; i < (int)centers.size() - 1; ++i) {
    path_finder_->reset();
    if (path_finder_->search(centers[i], centers[i + 1]) == Astar::REACH_END) {
      auto path = path_finder_->getPath();
      tour2.insert(tour2.end(), path.begin() + 1, path.end());
    } else {
      tour2.push_back(centers[i + 1]);
    }
  }
}

void HGrid::getFrontiersInGrid(const vector<int>& grid_ids, vector<int>& ftr_ids) {
  ftr_ids.clear();
  if (use_task_region_decision_) {
    std::unordered_map<int, int> uniq;
    for (int gid : grid_ids) {
      if (gid < 0 || gid >= (int)task_regions_.size()) continue;
      for (auto& p : task_regions_[gid].contained_frontier_ids_) uniq[p.first] = 1;
    }
    for (auto& kv : uniq) ftr_ids.push_back(kv.first);
    return;
  }
  int tmp = grid_ids.front();
  int grid_num1 = grid1_->grid_data_.size();
  if (tmp < grid_num1) {
    auto& grid = grid1_->grid_data_[tmp];
    for (auto pair : grid.contained_frontier_ids_) ftr_ids.push_back(pair.first);
  } else {
    tmp -= grid_num1;
    int coarse; fineToCoarseId(tmp, coarse);
    vector<int> fines; coarseToFineId(coarse, fines);
    vector<int> allocated_fines;
    for (auto fine : fines) {
      for (auto id : grid_ids) {
        if (fine + grid_num1 == id) { allocated_fines.push_back(fine); break; }
      }
    }
    for (auto fine : allocated_fines) {
      auto& grid = grid2_->grid_data_[fine];
      for (auto pair : grid.contained_frontier_ids_) ftr_ids.push_back(pair.first);
    }
  }
}

void HGrid::checkFirstGrid(const int& id) {
  auto& grid = getGrid(id);
  std::cout << "grid id: " << id << std::endl;
  std::cout << "unknown num: " << grid.unknown_num_ << std::endl;
  std::cout << "center: " << grid.center_.transpose() << std::endl;
  std::cout << "relevant: " << grid.is_cur_relevant_ << ", " << grid.is_prev_relevant_ << std::endl;
}

void HGrid::getGridMarker(vector<Eigen::Vector3d>& pts1, vector<Eigen::Vector3d>& pts2) {
  pts1.clear();
  pts2.clear();
  if (use_task_region_decision_) {
    auto map = edt_->sdf_map_;
    const double res = map->getResolution();
    for (int rid = 0; rid < (int)task_region_cells_.size(); ++rid) {
      auto& addr_set = task_region_addr_sets_[rid];
      for (auto& idx : task_region_cells_[rid]) {
        if (addr_set.find(map->toAddress(idx)) == addr_set.end()) continue;
        Eigen::Vector3d c; map->indexToPos(idx, c);
        double h = 0.5 * res;
        auto miss = [&](const Eigen::Vector3i& nidx) {
          return !map->isInMap(nidx) || addr_set.find(map->toAddress(nidx)) == addr_set.end();
        };
        if (miss(idx + Eigen::Vector3i(-1, 0, 0))) { pts1.emplace_back(c.x()-h, c.y()-h, 0.5); pts2.emplace_back(c.x()-h, c.y()+h, 0.5); }
        if (miss(idx + Eigen::Vector3i(1, 0, 0))) { pts1.emplace_back(c.x()+h, c.y()-h, 0.5); pts2.emplace_back(c.x()+h, c.y()+h, 0.5); }
        if (miss(idx + Eigen::Vector3i(0, -1, 0))) { pts1.emplace_back(c.x()-h, c.y()-h, 0.5); pts2.emplace_back(c.x()+h, c.y()-h, 0.5); }
        if (miss(idx + Eigen::Vector3i(0, 1, 0))) { pts1.emplace_back(c.x()-h, c.y()+h, 0.5); pts2.emplace_back(c.x()+h, c.y()+h, 0.5); }
      }
    }
    return;
  }
  for (auto& grid : grid1_->grid_data_) {
    if (!grid.active_) continue;
    for (int i = 0; i < 4; ++i) { pts1.push_back(grid.vertices_[i]); pts2.push_back(grid.vertices_[(i + 1) % 4]); }
  }
  for (auto& grid : grid2_->grid_data_) {
    if (!grid.active_) continue;
    for (int i = 0; i < 4; ++i) { pts1.push_back(grid.vertices_[i]); pts2.push_back(grid.vertices_[(i + 1) % 4]); }
  }
  for (auto& pt : pts1) pt[2] = 0.5;
  for (auto& pt : pts2) pt[2] = 0.5;
}

void HGrid::getGridMarker2(vector<Eigen::Vector3d>& pts, vector<string>& texts) {
  pts.clear();
  texts.clear();
  if (use_task_region_decision_) {
    for (int i = 0; i < (int)task_regions_.size(); ++i) { pts.push_back(task_regions_[i].center_); texts.push_back(std::string("R") + std::to_string(i)); }
    return;
  }
  for (int i = 0; i < (int)grid1_->grid_data_.size(); ++i) {
    auto& grid = grid1_->grid_data_[i];
    if (!grid.active_ || !grid.is_cur_relevant_) continue;
    pts.push_back(grid.center_); texts.push_back(to_string(i));
  }
  const int grid_num1 = grid1_->grid_data_.size();
  for (int i = 0; i < (int)grid2_->grid_data_.size(); ++i) {
    auto& grid = grid2_->grid_data_[i];
    if (!grid.active_ || !grid.is_cur_relevant_) continue;
    pts.push_back(grid.center_); texts.push_back(to_string(i + grid_num1));
  }
}

double HGrid::getMinDist2DToBoundary1(const Eigen::Vector3d& pos)//自己加的
{//只确定距离最小的边界距离，应该确定距离最小的角落距离
  Eigen::Vector3d ori, size;
  edt_->sdf_map_->getRegion(ori, size);
  Eigen::Vector2d min_xy(ori(0), ori(1));
  Eigen::Vector2d max_xy = min_xy + Eigen::Vector2d(size(0), size(1));
  Eigen::Vector2d p_xy(pos(0), pos(1));

  double left   = p_xy.x() - min_xy.x();
  double right  = max_xy.x() - p_xy.x();
  double bottom = p_xy.y() - min_xy.y();
  double top    = max_xy.y() - p_xy.y();

  return std::min({left, right}); 
}
double HGrid::getMinDist2DToBoundary2(const Eigen::Vector3d& pos)//自己加的
{//只确定距离最小的边界距离，应该确定距离最小的角落距离
  Eigen::Vector3d ori, size;
  edt_->sdf_map_->getRegion(ori, size);
  Eigen::Vector2d min_xy(ori(0), ori(1));
  Eigen::Vector2d max_xy = min_xy + Eigen::Vector2d(size(0), size(1));
  Eigen::Vector2d p_xy(pos(0), pos(1));

  double left   = p_xy.x() - min_xy.x();
  double right  = max_xy.x() - p_xy.x();
  double bottom = p_xy.y() - min_xy.y();
  double top    = max_xy.y() - p_xy.y();

  return std::min({left, right, bottom, top}); 
}

}  // namespace fast_planner