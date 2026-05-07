#ifndef _HGRID_H_
#define _HGRID_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <algorithm>

using Eigen::Vector3d;
using std::list;
using std::pair;
using std::shared_ptr;
using std::unique_ptr;
using std::unordered_map;
using std::unordered_set;
using std::vector;

class RayCaster;

namespace fast_planner {

class EDTEnvironment;
class Astar;
class GridInfo;
class UniformGrid;

class HGrid {

public:
  HGrid(const shared_ptr<EDTEnvironment>& edt, ros::NodeHandle& nh);
  ~HGrid();

  void updateGridData(const int& drone_id, vector<int>& grid_ids, bool reallocated,
      const vector<int>& last_grid_ids, vector<int>& first_ids, vector<int>& second_ids);

  bool updateBaseCoor();
  void inputFrontiers(const vector<Eigen::Vector3d>& avgs);
  void buildTaskRegions();
  bool shouldRebuildTaskRegions(const vector<Eigen::Vector3d>& avgs) const;
  void getCostMatrix(const vector<Eigen::Vector3d>& positions,
      const vector<Eigen::Vector3d>& velocities, const vector<vector<int>>& first_ids,
      const vector<vector<int>>& second_ids, const vector<int>& grid_ids, Eigen::MatrixXd& mat);
  void getGridTour(const vector<int>& ids, const Eigen::Vector3d& pos,
      vector<Eigen::Vector3d>& tour, vector<Eigen::Vector3d>& tour2);
  void getFrontiersInGrid(const vector<int>& grid_ids, vector<int>& ftr_ids);
  bool getNextGrid(const vector<int>& grid_ids, Eigen::Vector3d& grid_pos, double& grid_yaw);
  void getConsistentGrid(const vector<int>& last_ids, const vector<int>& cur_ids,
      vector<int>& first_ids, vector<int>& second_ids);

  void getGridMarker(vector<Eigen::Vector3d>& pts1, vector<Eigen::Vector3d>& pts2);
  void getGridMarker2(vector<Eigen::Vector3d>& pts, vector<std::string>& texts);
  void checkFirstGrid(const int& id);
  int getUnknownCellsNum(const int& grid_id);
  int getRegionTotalCellsNum(const int& grid_id);
  double getRegionUnknownRatio(const int& grid_id);
  Eigen::Vector3d getCenter(const int& grid_id);
  void getActiveGrids(vector<int>& grid_ids);
  bool isConsistent(const int& id1, const int& id2);
  bool isTaskRegionMode() const { return use_task_region_decision_; }
  int getRegionGroupId(const int& id) const;
  double getCostDroneToGrid(
      const Eigen::Vector3d& pos, const int& grid_id, const vector<int>& first);
  double getCostGridToGrid(const int& id1, const int& id2, const vector<vector<int>>& firsts,
      const vector<vector<int>>& seconds, const int& drone_num);
  double getMinDist2DToBoundary1(const Eigen::Vector3d& pos);
  double getMinDist2DToBoundary2(const Eigen::Vector3d& pos);
  unique_ptr<Astar> path_finder_;

private:
  bool use_task_region_decision_;
  vector<Eigen::Vector3d> frontier_avgs_;
  vector<GridInfo> task_regions_;
  vector<vector<Eigen::Vector3i>> task_region_cells_;
  vector<unordered_set<int>> task_region_addr_sets_;
  vector<Eigen::Vector2d> task_region_anchors_;
  vector<Eigen::Vector2d> task_region_tangents_;
  vector<int> task_region_group_ids_;

  double task_cluster_radius_, task_band_radius_, task_region_inflate_xy_, task_z_tol_;
  double task_axis_neighbor_radius_, task_half_span_, task_seed_radius_, task_overlap_bias_;
  double task_rebuild_min_interval_, task_rebuild_frontier_shift_, task_anchor_resolution_;
  int task_min_frontiers_, task_max_frontiers_per_region_, task_min_free_cells_,
      task_min_unknown_cells_, task_max_anchor_regions_;

  mutable ros::Time last_task_region_build_time_;
  mutable int last_frontier_count_ = -1;
  mutable Eigen::Vector3d last_frontier_centroid_ = Eigen::Vector3d::Zero();

  void coarseToFineId(const int& coarse, vector<int>& fines);
  void fineToCoarseId(const int& fine, int& coarse);
  GridInfo& getGrid(const int& id);

  bool isClose(const int& id1, const int& id2);
  bool inSameLevel1(const int& id1, const int& id2);

  unique_ptr<UniformGrid> grid1_;  // Coarse level
  unique_ptr<UniformGrid> grid2_;  // Fine level

  shared_ptr<EDTEnvironment> edt_;
  double consistent_cost_;
  double consistent_cost2_;

  Eigen::Matrix3d rot_sw_;
  Eigen::Vector3d trans_sw_;
  bool use_swarm_tf_;
  double w_first_;
};

}  // namespace fast_planner
#endif
