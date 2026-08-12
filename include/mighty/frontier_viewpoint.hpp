/* ----------------------------------------------------------------------------
 * Perception-aware frontier viewpoint selector (ground rover, curb/negative-obstacle
 * safety). Decoupled, ROS-free geometry core:
 *
 *   frontier cluster cells --PCA--> (tangent t, UNKNOWN-facing normal n)
 *      --> tiny 1-D lateral search of poses q(s) = C - d*n + s*t
 *          subject to HARD footprint-free + ESDF + Mid-360 potential-visibility
 *      --> smallest |s| valid, with a desired observation yaw facing the unknown.
 *
 * The map is accessed only through GridQuery callbacks (world coords), so this file
 * depends on nothing but Eigen and the STL and is unit-testable without ROS/MIGHTY.
 * The purpose is NOT to infer the hidden terrain -- only to prefer a nearby safe
 * diagonal pose that could geometrically observe a curb/step behind the frontier,
 * instead of blindly driving to the FREE/UNKNOWN boundary centroid.
 * -------------------------------------------------------------------------- */
#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

namespace mighty {

// ---------------------------------------------------------------------------
// Parameters (mirrors config exploration.viewpoint.*; see mighty_ground_robot.yaml)
// ---------------------------------------------------------------------------
struct ViewpointParams {
  bool   enabled                 = true;
  double standoff_m              = 0.75;   // perpendicular center->viewpoint distance d
  double lateral_step_m          = 0.15;   // tangent search step (~map res)
  double max_lateral_offset_m    = 1.50;   // max |s|
  double footprint_margin_m      = 0.10;   // added to r_footprint
  double min_esdf_clearance_m    = 0.70;   // required if an ESDF is available
  double pca_min_anisotropy      = 1.5;    // lambda_max/lambda_min below this => invalid
  double normal_probe_m          = 0.30;   // probe distance for normal-sign resolution
  std::vector<double> target_depths_m = {0.25, 0.40, 0.55, 0.70};  // g_j = C + depth*n
  double critical_drop_m         = 0.15;   // hypothetical downward step H
  double min_visible_fraction    = 0.50;   // accept if visible/total >= this
  // Mid-360 model (config fallback; see Part 4 -- URDF sim pitch 0.3 rad, HW ~20 deg).
  double sensor_height_m         = 0.51;
  double sensor_mount_pitch_deg  = 20.0;   // nose-down forward pitch
  double sensor_vertical_min_deg = -7.0;   // native band
  double sensor_vertical_max_deg = 52.0;
  bool   fallback_to_legacy_centroid = false;
  // Approach-heading: a pre-viewpoint q_pre = q* - l*hat(g-q*) so the FINAL path segment
  // q_pre -> q* points at the target (the executed heading comes from that segment via
  // the path-direction yaw behavior -- NOT from a fake in-place turn, which the hardware
  // controller cannot command). The segment must be footprint/ESDF safe.
  double pre_viewpoint_len_m     = 0.50;   // l [m]
  double arrival_tol_m           = 0.22;   // dedicated viewpoint arrival tolerance (node)
  // Reveal test (post-dwell): a PCA-aligned strip BEHIND the frontier is snapshotted for
  // UNKNOWN cells at selection; after the dwell, R = 1 - N_unknown_after/N_unknown_before.
  // R >= min_reveal_fraction => the VIEWPOINT succeeded (this does NOT by itself mark the
  // frontier VISITED -- updated WFD/matching decides residual vs resolved).
  double min_reveal_fraction     = 0.30;
  double strip_half_width_m      = 0.75;   // lateral (tangent) half-extent of the strip
  double strip_depth_m           = 0.0;    // along-normal depth; <=0 => max(target_depths_m)
  // Footprint radius from the configured XY bounding box: r = 0.5*sqrt(bx^2+by^2).
  double robot_bbox_x            = 0.6;
  double robot_bbox_y            = 0.6;

  double rFootprint() const { return 0.5 * std::sqrt(robot_bbox_x * robot_bbox_x +
                                                     robot_bbox_y * robot_bbox_y); }
};

// ---------------------------------------------------------------------------
// Map access (world coords). ESDF callbacks may be empty (=> ESDF check skipped).
// ---------------------------------------------------------------------------
struct GridQuery {
  std::function<bool(double, double)> isFree;      // known-FREE at world (x,y)
  std::function<bool(double, double)> isUnknown;   // UNKNOWN at world (x,y)
  std::function<bool(double, double)> isOccupied;  // OCCUPIED at world (x,y)
  double resolution = 0.15;                        // grid resolution [m]
  std::function<bool(double, double)>   esdfInBounds;  // optional
  std::function<double(double, double)> esdfDistance;  // optional (only if in bounds)
};

// ---------------------------------------------------------------------------
// PCA geometry of one frontier
// ---------------------------------------------------------------------------
struct FrontierGeometry {
  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  Eigen::Vector2d tangent  = Eigen::Vector2d::Zero();  // unit, largest-eigenvalue axis
  Eigen::Vector2d normal   = Eigen::Vector2d::Zero();  // unit, +n points toward UNKNOWN
  double anisotropy = 0.0;   // lambda_max / lambda_min (>= 1)
  bool   valid      = false; // false => too compact/curved for a stable direction
};

enum class ViewpointReject {
  NONE = 0,
  PCA_INVALID,
  FOOTPRINT_UNKNOWN,
  FOOTPRINT_OCCUPIED,
  LOW_CLEARANCE,
  LOW_VISIBILITY,
  APPROACH_UNSAFE,   // q_pre -> q* terminal segment not footprint/ESDF safe
  NO_CANDIDATE,
};
const char* viewpointRejectStr(ViewpointReject r);

// One valid viewpoint pose (the node tries these in order as alternates).
struct ViewpointPose {
  Eigen::Vector2d q      = Eigen::Vector2d::Zero();
  Eigen::Vector2d q_pre  = Eigen::Vector2d::Zero();
  double s               = 0.0;
  double yaw             = 0.0;
  double clearance_m     = 0.0;
  int    visible_samples = 0;
  std::vector<Eigen::Vector2d> target_samples;  // visible g_j (pre-nav FOV/lip checks)
};

struct ViewpointResult {
  bool   ok = false;
  Eigen::Vector2d q      = Eigen::Vector2d::Zero();  // chosen viewpoint (world) == ranked[0]
  Eigen::Vector2d q_pre  = Eigen::Vector2d::Zero();  // pre-viewpoint; final leg q_pre->q
  double s               = 0.0;                       // chosen lateral offset
  double yaw             = 0.0;                        // desired observation yaw [rad]
  Eigen::Vector2d target = Eigen::Vector2d::Zero();    // representative unknown target
  std::vector<Eigen::Vector2d> target_samples;        // hypothetical target points g_j (world)
  std::vector<ViewpointPose> ranked;                  // ALL valid poses, best-first (alternates)
  int    visible_samples = 0;
  int    total_samples   = 0;
  double clearance_m     = 0.0;                        // ESDF clearance at q (or +inf)
  ViewpointReject reason = ViewpointReject::NO_CANDIDATE;
  FrontierGeometry geom;
  // Per-candidate audit for logging/RViz: (s, valid, reject-reason).
  std::vector<std::tuple<double, bool, ViewpointReject>> candidates;
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

/** @brief PCA on frontier cells -> (tangent, UNKNOWN-facing normal, anisotropy, valid).
 *  PCA gives the boundary orientation; the occupancy grid only resolves the normal
 *  SIGN (probe +/- normal_probe_m and pick the side with more UNKNOWN support). If
 *  @p prev is non-null, the new tangent/normal signs are aligned to it before return
 *  (frame-to-frame stability). Marks valid=false when anisotropy < pca_min_anisotropy. */
FrontierGeometry computeFrontierGeometry(const std::vector<Eigen::Vector2d>& cells,
                                         const GridQuery& grid, const ViewpointParams& P,
                                         const FrontierGeometry* prev = nullptr);

/** @brief Given a frontier centroid + geometry, search q(s) = C - d*n + s*t over the
 *  smallest |s| that is footprint-free, ESDF-clear, and can potentially observe the
 *  hypothetical targets behind the frontier (Mid-360 FOV + curb-lip + 2-D occlusion).
 *  Deterministic: minimize |s|; +s/-s tie -> closer to robot -> greater clearance. */
ViewpointResult selectViewpoint(const Eigen::Vector2d& centroid, const FrontierGeometry& geom,
                                const Eigen::Vector2d& robot_xy, const GridQuery& grid,
                                const ViewpointParams& P);

// ---- Exposed helpers (unit-tested directly) --------------------------------

/** @brief Is the whole disc of radius r_safe = rFootprint()+margin around q known FREE?
 *  Returns NONE if free, else FOOTPRINT_UNKNOWN / FOOTPRINT_OCCUPIED (OCC dominates). */
ViewpointReject footprintCheck(const Eigen::Vector2d& q, const GridQuery& grid,
                               const ViewpointParams& P);

/** @brief Is the terminal approach segment a->b footprint-free and (if ESDF present)
 *  clearance-safe along its length? Samples the segment at grid resolution and runs
 *  footprintCheck + the ESDF clearance test at each sample. */
bool segmentFootprintSafe(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                          const GridQuery& grid, const ViewpointParams& P);

/** @brief Potential visibility of a single hypothetical target g behind the frontier,
 *  from viewpoint q (sensor faces the frontier). Combines the curb-lip clearance
 *  (x >= H*d/h) with the Mid-360 vertical-FOV depression window and (optional) 2-D
 *  occupancy occlusion along q->frontier. @p depth is the along-normal distance of g
 *  behind the frontier; @p standoff is the perpendicular d. */
bool targetPotentiallyVisible(const Eigen::Vector2d& q, const Eigen::Vector2d& g,
                              double depth, double standoff, const GridQuery& grid,
                              const ViewpointParams& P);

/** @brief Snapshot the world centers of all UNKNOWN cells inside a PCA-aligned strip
 *  behind the frontier: along +normal over [0, strip_depth] and along +/-tangent over
 *  [-strip_half_width, +strip_half_width]. Iterates only the strip's local bounding box
 *  (not the whole grid). Used to measure the real reveal after the observation dwell. */
std::vector<Eigen::Vector2d> snapshotTargetStripUnknown(const Eigen::Vector2d& centroid,
                                                        const FrontierGeometry& geom,
                                                        const GridQuery& grid,
                                                        const ViewpointParams& P);

/** @brief Reveal fraction R = 1 - N_unknown_after/N_unknown_before over the snapshotted
 *  cells (how many stopped being UNKNOWN). Returns 1.0 if the snapshot was empty. */
double revealFraction(const std::vector<Eigen::Vector2d>& snapshot_unknown, const GridQuery& grid);

}  // namespace mighty
