/* ----------------------------------------------------------------------------
 * Perception-aware frontier viewpoint selector -- geometry core.
 * See include/mighty/frontier_viewpoint.hpp for the model.
 * -------------------------------------------------------------------------- */
#include "mighty/frontier_viewpoint.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mighty {

namespace {
constexpr double kDeg2Rad = M_PI / 180.0;

// Count UNKNOWN vs FREE support a short distance along +dir from centroid c.
int unknownSupportAlong(const Eigen::Vector2d& c, const Eigen::Vector2d& dir, double probe,
                        const GridQuery& grid) {
  // Sample a few points out to `probe` along dir; +1 per UNKNOWN, -1 per FREE.
  int score = 0;
  const int n = 3;
  for (int k = 1; k <= n; ++k) {
    const double r = probe * (static_cast<double>(k) / n);
    const Eigen::Vector2d p = c + r * dir;
    if (grid.isUnknown && grid.isUnknown(p.x(), p.y())) ++score;
    else if (grid.isFree && grid.isFree(p.x(), p.y())) --score;
  }
  return score;
}
}  // namespace

const char* viewpointRejectStr(ViewpointReject r) {
  switch (r) {
    case ViewpointReject::NONE: return "none";
    case ViewpointReject::PCA_INVALID: return "pca_invalid";
    case ViewpointReject::FOOTPRINT_UNKNOWN: return "footprint_unknown";
    case ViewpointReject::FOOTPRINT_OCCUPIED: return "footprint_occupied";
    case ViewpointReject::LOW_CLEARANCE: return "low_clearance";
    case ViewpointReject::LOW_VISIBILITY: return "insufficient_visibility";
    case ViewpointReject::APPROACH_UNSAFE: return "approach_unsafe";
    case ViewpointReject::NO_TARGETS: return "no_targets";
    case ViewpointReject::NO_CANDIDATE: return "no_candidate";
  }
  return "?";
}

// ===========================================================================
// PCA
// ===========================================================================
FrontierGeometry computeFrontierGeometry(const std::vector<Eigen::Vector2d>& cells,
                                         const GridQuery& grid, const ViewpointParams& P,
                                         const FrontierGeometry* prev) {
  FrontierGeometry g;
  const int n = static_cast<int>(cells.size());
  if (n < 3) return g;  // too few cells for a meaningful covariance

  // mu and 2x2 covariance C = mean((p-mu)(p-mu)^T).
  Eigen::Vector2d mu = Eigen::Vector2d::Zero();
  for (const auto& p : cells) mu += p;
  mu /= static_cast<double>(n);
  double sxx = 0.0, sxy = 0.0, syy = 0.0;
  for (const auto& p : cells) {
    const double dx = p.x() - mu.x(), dy = p.y() - mu.y();
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  sxx /= n;
  sxy /= n;
  syy /= n;
  g.centroid = mu;

  // Closed-form symmetric 2x2 eigendecomposition.
  const double tr = sxx + syy;
  const double det = sxx * syy - sxy * sxy;
  double disc = tr * tr / 4.0 - det;
  if (disc < 0.0) disc = 0.0;
  const double sq = std::sqrt(disc);
  const double l_max = tr / 2.0 + sq;
  const double l_min = tr / 2.0 - sq;

  // Eigenvector for l_max = frontier tangent.
  Eigen::Vector2d t;
  if (std::abs(sxy) > 1e-12) {
    t = Eigen::Vector2d(l_max - syy, sxy);
  } else {
    t = (sxx >= syy) ? Eigen::Vector2d(1.0, 0.0) : Eigen::Vector2d(0.0, 1.0);
  }
  if (t.norm() < 1e-12) t = Eigen::Vector2d(1.0, 0.0);
  t.normalize();
  Eigen::Vector2d nrm(-t.y(), t.x());  // perpendicular = small-eigenvalue axis

  g.tangent = t;
  g.normal = nrm;
  g.anisotropy = (l_min > 1e-12) ? (l_max / l_min) : std::numeric_limits<double>::infinity();

  // Degeneracy: too compact/curved => no stable dominant direction.
  if (!(g.anisotropy >= P.pca_min_anisotropy)) {
    g.valid = false;
    return g;
  }

  // Resolve the normal SIGN from occupancy: +n must point toward UNKNOWN.
  const int sup_plus = unknownSupportAlong(mu, g.normal, P.normal_probe_m, grid);
  const int sup_minus = unknownSupportAlong(mu, -g.normal, P.normal_probe_m, grid);
  if (sup_minus > sup_plus) g.normal = -g.normal;

  // Frame-to-frame stability: align signs to the previous estimate if given.
  if (prev && prev->valid) {
    if (g.tangent.dot(prev->tangent) < 0.0) g.tangent = -g.tangent;
    if (g.normal.dot(prev->normal) < 0.0) g.normal = -g.normal;
  }

  g.valid = true;
  return g;
}

// ===========================================================================
// Footprint safety
// ===========================================================================
ViewpointReject footprintCheck(const Eigen::Vector2d& q, const GridQuery& grid,
                               const ViewpointParams& P) {
  const double r_safe = P.rFootprint() + P.footprint_margin_m;
  const double r2 = r_safe * r_safe;
  const double step = std::max(1e-3, grid.resolution);
  bool any_occ = false, any_unk = false;
  // Always check the cell under q (robust when r_safe < grid.resolution, where the
  // disc sampling below could otherwise step over every cell center).
  if (grid.isOccupied && grid.isOccupied(q.x(), q.y())) any_occ = true;
  else if (grid.isFree && !grid.isFree(q.x(), q.y())) any_unk = true;
  for (double dy = -r_safe; dy <= r_safe + 1e-9; dy += step) {
    for (double dx = -r_safe; dx <= r_safe + 1e-9; dx += step) {
      if (dx * dx + dy * dy > r2) continue;  // cell center outside the disc
      const double wx = q.x() + dx, wy = q.y() + dy;
      if (grid.isOccupied && grid.isOccupied(wx, wy)) any_occ = true;
      else if (grid.isFree && !grid.isFree(wx, wy)) any_unk = true;  // not-free & not-occ => UNK/OOB
    }
  }
  if (any_occ) return ViewpointReject::FOOTPRINT_OCCUPIED;  // OCC dominates
  if (any_unk) return ViewpointReject::FOOTPRINT_UNKNOWN;
  return ViewpointReject::NONE;
}

ViewpointReject pathFootprintKnownFree(const std::vector<Eigen::Vector2d>& path,
                                       const GridQuery& grid, const ViewpointParams& P) {
  if (path.size() < 2) return path.empty() ? ViewpointReject::NO_CANDIDATE : ViewpointReject::NONE;
  const double step = std::max(1e-3, grid.resolution);  // sample spacing <= grid resolution
  bool any_unk = false;
  for (size_t i = 1; i < path.size(); ++i) {
    const Eigen::Vector2d a = path[i - 1], b = path[i];
    const double len = (b - a).norm();
    const int ns = std::max(1, static_cast<int>(std::ceil(len / step)));
    for (int k = 0; k <= ns; ++k) {
      const double u = static_cast<double>(k) / ns;
      const Eigen::Vector2d p = a + u * (b - a);
      const ViewpointReject fp = footprintCheck(p, grid, P);  // full footprint disc, occ-only
      if (fp == ViewpointReject::FOOTPRINT_OCCUPIED) return fp;  // OCC dominates -> fail now
      if (fp == ViewpointReject::FOOTPRINT_UNKNOWN) any_unk = true;
    }
  }
  return any_unk ? ViewpointReject::FOOTPRINT_UNKNOWN : ViewpointReject::NONE;
}

bool segmentFootprintSafe(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const GridQuery& grid,
                          const ViewpointParams& P) {
  const double len = (b - a).norm();
  const int nsteps = std::max(1, static_cast<int>(std::ceil(len / std::max(1e-3, grid.resolution))));
  for (int k = 0; k <= nsteps; ++k) {
    const double u = static_cast<double>(k) / nsteps;
    const Eigen::Vector2d p = a + u * (b - a);
    if (footprintCheck(p, grid, P) != ViewpointReject::NONE) return false;
    if (grid.esdfInBounds && grid.esdfDistance) {
      if (!grid.esdfInBounds(p.x(), p.y())) return false;
      if (grid.esdfDistance(p.x(), p.y()) < P.min_esdf_clearance_m) return false;
    }
  }
  return true;
}

// ===========================================================================
// Single-target potential visibility (Mid-360 FOV + curb-lip + 2-D occlusion)
// ===========================================================================
bool targetPotentiallyVisible(const Eigen::Vector2d& q, const Eigen::Vector2d& g, double depth,
                              double standoff, const GridQuery& grid, const ViewpointParams& P,
                              double psi_obs, const SensorExtrinsics& extr) {
  const double h = P.sensor_height_m;
  const double H = P.critical_drop_m;
  const double d = standoff;

  // (a) Curb-lip clearance: a lower-ground target at depth x behind a straight lip of
  // height H, seen from perpendicular distance d with sensor height h, is occluded
  // until x >= H*d/h. The lateral offset does not remove straight-lip occlusion.
  if (h > 1e-6 && depth < (H * d / h)) return false;

  // (b) Vertical-FOV window using the ACTUAL base->lidar orientation. The map-frame ray
  // from the sensor to the (lower-ground) target is the horizontal (g - q) with vertical
  // drop -(h + H); rotate it into the LiDAR frame via Rz(psi_obs) * R_base_lidar and
  // take its elevation. When extr is valid this is TF-driven; otherwise it falls back to
  // the scalar sensor_mount_pitch_deg about +Y (which reproduces the old p - delta).
  const Eigen::Vector2d dir = g - q;
  const Eigen::Vector3d ray_map(dir.x(), dir.y(), -(h + H));
  const Eigen::Matrix3d R_base_lidar =
      extr.valid ? extr.R_base_lidar
                 : Eigen::Matrix3d(Eigen::AngleAxisd(P.sensor_mount_pitch_deg * kDeg2Rad,
                                                     Eigen::Vector3d::UnitY()));
  const Eigen::Matrix3d R_map_lidar =
      Eigen::Matrix3d(Eigen::AngleAxisd(psi_obs, Eigen::Vector3d::UnitZ())) * R_base_lidar;
  const Eigen::Vector3d p_lidar = R_map_lidar.transpose() * ray_map;  // map->lidar
  const double elev = std::atan2(p_lidar.z(), std::hypot(p_lidar.x(), p_lidar.y()));
  const double vmin = P.sensor_vertical_min_deg * kDeg2Rad;
  const double vmax = P.sensor_vertical_max_deg * kDeg2Rad;
  if (elev < vmin || elev > vmax) return false;

  // (c) Cheap 2-D occlusion: reject if a known-OCCUPIED cell blocks the segment q->g.
  if (grid.isOccupied) {
    const double dist = (g - q).norm();
    const int nsteps = std::max(1, static_cast<int>(std::ceil(dist / std::max(1e-3, grid.resolution))));
    for (int k = 1; k < nsteps; ++k) {  // skip endpoints
      const double u = static_cast<double>(k) / nsteps;
      const Eigen::Vector2d s = q + u * (g - q);
      if (grid.isOccupied(s.x(), s.y())) return false;
    }
  }
  return true;
}

// ===========================================================================
// Target-strip snapshot + reveal fraction
// ===========================================================================
std::vector<Eigen::Vector2d> snapshotTargetStripUnknown(const Eigen::Vector2d& centroid,
                                                        const FrontierGeometry& geom,
                                                        const GridQuery& grid,
                                                        const ViewpointParams& P) {
  std::vector<Eigen::Vector2d> out;
  if (!geom.valid || !grid.isUnknown) return out;

  double depth = P.strip_depth_m;
  if (depth <= 0.0) {
    for (double d : P.target_depths_m) depth = std::max(depth, d);
    if (depth <= 0.0) depth = P.standoff_m;
  }
  const double half_w = P.strip_half_width_m;
  const double res = std::max(1e-3, grid.resolution);
  const Eigen::Vector2d n = geom.normal, t = geom.tangent;

  // Local bounding box of the rotated strip corners (scan only this box).
  double minx = 1e18, miny = 1e18, maxx = -1e18, maxy = -1e18;
  for (double a : {0.0, depth}) {
    for (double b : {-half_w, half_w}) {
      const Eigen::Vector2d c = centroid + a * n + b * t;
      minx = std::min(minx, c.x()); maxx = std::max(maxx, c.x());
      miny = std::min(miny, c.y()); maxy = std::max(maxy, c.y());
    }
  }
  for (double y = miny; y <= maxy + 1e-9; y += res) {
    for (double x = minx; x <= maxx + 1e-9; x += res) {
      const Eigen::Vector2d rel(x - centroid.x(), y - centroid.y());
      const double an = rel.dot(n);  // along +normal (behind the frontier)
      const double at = rel.dot(t);  // along tangent
      if (an < 0.0 || an > depth) continue;
      if (std::abs(at) > half_w) continue;
      if (grid.isUnknown(x, y)) out.emplace_back(x, y);
    }
  }
  return out;
}

double revealFraction(const std::vector<Eigen::Vector2d>& snapshot_unknown, const GridQuery& grid) {
  // An empty target strip carries NO observation evidence -> not a successful reveal.
  // The caller detects empty separately (EMPTY_TARGET_STRIP) and releases the frontier.
  if (snapshot_unknown.empty()) return 0.0;
  if (!grid.isUnknown) return 0.0;
  int still = 0;
  for (const auto& c : snapshot_unknown)
    if (grid.isUnknown(c.x(), c.y())) ++still;
  return 1.0 - static_cast<double>(still) / static_cast<double>(snapshot_unknown.size());
}

// ===========================================================================
// Viewpoint selection (tiny 1-D search along the tangent)
// ===========================================================================
namespace {
// One candidate q(s). Fills q, q_pre, visible/total/clearance, and the visible target
// samples; returns the reject reason (NONE = valid). `target` is the representative
// unknown point used for the approach heading (hat(target - q)).
ViewpointReject evalCandidate(double s, const Eigen::Vector2d& C, const FrontierGeometry& geom,
                              const Eigen::Vector2d& target, const GridQuery& grid,
                              const ViewpointParams& P, const SensorExtrinsics& extr,
                              Eigen::Vector2d& q_out, Eigen::Vector2d& q_pre_out, int& vis_out,
                              int& tot_out, double& clear_out,
                              std::vector<Eigen::Vector2d>& samples_out) {
  const Eigen::Vector2d q = C - P.standoff_m * geom.normal + s * geom.tangent;
  q_out = q;
  const double psi_obs = std::atan2(target.y() - q.y(), target.x() - q.x());  // rover faces target
  q_pre_out = q;
  vis_out = 0;
  tot_out = static_cast<int>(P.target_depths_m.size());
  clear_out = std::numeric_limits<double>::infinity();
  samples_out.clear();

  const ViewpointReject fp = footprintCheck(q, grid, P);
  if (fp != ViewpointReject::NONE) return fp;

  if (grid.esdfInBounds && grid.esdfDistance) {
    if (!grid.esdfInBounds(q.x(), q.y())) return ViewpointReject::LOW_CLEARANCE;  // unknown clearance
    clear_out = grid.esdfDistance(q.x(), q.y());
    if (clear_out < P.min_esdf_clearance_m) return ViewpointReject::LOW_CLEARANCE;
  }

  for (double depth : P.target_depths_m) {
    const Eigen::Vector2d g = C + depth * geom.normal;
    if (targetPotentiallyVisible(q, g, depth, P.standoff_m, grid, P, psi_obs, extr)) {
      ++vis_out;
      samples_out.push_back(g);
    }
  }
  if (tot_out == 0 || static_cast<double>(vis_out) / tot_out < P.min_visible_fraction) {
    return ViewpointReject::LOW_VISIBILITY;
  }

  // Approach-heading: pre-viewpoint q_pre = q - l * hat(target - q). The FINAL executed
  // path segment q_pre -> q must be safe so the path-direction yaw at arrival faces the
  // target (no in-place turn is commanded).
  Eigen::Vector2d hdir = target - q;
  if (hdir.norm() < 1e-6) hdir = geom.normal;  // degenerate: fall back to +n
  hdir.normalize();
  const Eigen::Vector2d q_pre = q - P.pre_viewpoint_len_m * hdir;
  if (!segmentFootprintSafe(q_pre, q, grid, P)) return ViewpointReject::APPROACH_UNSAFE;
  q_pre_out = q_pre;
  return ViewpointReject::NONE;
}
}  // namespace

ViewpointResult selectViewpoint(const Eigen::Vector2d& centroid, const FrontierGeometry& geom,
                                const Eigen::Vector2d& robot_xy, const GridQuery& grid,
                                const ViewpointParams& P, const SensorExtrinsics& extr) {
  ViewpointResult res;
  res.geom = geom;
  if (!geom.valid) {
    res.reason = ViewpointReject::PCA_INVALID;
    return res;
  }

  const int max_k = std::max(0, static_cast<int>(std::floor(P.max_lateral_offset_m /
                                                            std::max(1e-6, P.lateral_step_m))));

  // Representative unknown target (fixed for all s): centroid pushed the mean target
  // depth into the unknown. Used for the observation yaw and the approach direction.
  double mean_depth = P.standoff_m;
  if (!P.target_depths_m.empty()) {
    double acc = 0.0;
    for (double d : P.target_depths_m) acc += d;
    mean_depth = acc / P.target_depths_m.size();
  }
  const Eigen::Vector2d target = centroid + mean_depth * geom.normal;
  res.target = target;

  // Evaluate every candidate s = 0, +/-step, ..., collect ALL valid poses, then rank
  // them so the node can try alternates (best-first) if the reveal test later fails.
  struct Cand {
    ViewpointPose pose;
    double mag;
    double dist;
  };
  std::vector<Cand> valid;
  for (int k = 0; k <= max_k; ++k) {
    const double mag = k * P.lateral_step_m;
    double offs[2];
    int noff = 0;
    if (k == 0) {
      offs[noff++] = 0.0;
    } else {
      offs[noff++] = +mag;
      offs[noff++] = -mag;
    }
    for (int i = 0; i < noff; ++i) {
      Eigen::Vector2d q, q_pre;
      int vis, tot;
      double clr;
      std::vector<Eigen::Vector2d> samples;
      const ViewpointReject r = evalCandidate(offs[i], centroid, geom, target, grid, P, extr, q,
                                              q_pre, vis, tot, clr, samples);
      res.candidates.emplace_back(offs[i], r == ViewpointReject::NONE, r);
      if (r != ViewpointReject::NONE) {
        if (res.reason == ViewpointReject::NO_CANDIDATE) res.reason = r;  // first failure reason
        continue;
      }
      ViewpointPose vp;
      vp.q = q;
      vp.q_pre = q_pre;
      vp.s = offs[i];
      vp.clearance_m = clr;
      vp.visible_samples = vis;
      vp.target_samples = std::move(samples);
      vp.yaw = std::atan2(target.y() - q.y(), target.x() - q.x());
      valid.push_back({vp, std::abs(offs[i]), (q - robot_xy).norm()});
    }
  }

  if (valid.empty()) {  // res.reason holds the first observed failure
    if (res.reason == ViewpointReject::NONE) res.reason = ViewpointReject::NO_CANDIDATE;
    return res;
  }

  // Rank: smallest |s|, then closer to the robot, then greater clearance.
  std::sort(valid.begin(), valid.end(), [](const Cand& a, const Cand& b) {
    if (std::abs(a.mag - b.mag) > 1e-9) return a.mag < b.mag;
    if (std::abs(a.dist - b.dist) > 1e-9) return a.dist < b.dist;
    return a.pose.clearance_m > b.pose.clearance_m;
  });
  res.ranked.reserve(valid.size());
  for (auto& c : valid) res.ranked.push_back(std::move(c.pose));

  const ViewpointPose& best = res.ranked.front();
  res.ok = true;
  res.q = best.q;
  res.q_pre = best.q_pre;
  res.s = best.s;
  res.yaw = best.yaw;
  res.clearance_m = best.clearance_m;
  res.visible_samples = best.visible_samples;
  res.total_samples = static_cast<int>(P.target_depths_m.size());
  res.target_samples = best.target_samples;
  res.reason = ViewpointReject::NONE;
  return res;
}

// ===========================================================================
// v3: near-ground blind mask
// ===========================================================================
BlindMask BlindMask::build(const Eigen::Matrix3d& R_base_lidar, double h, int n_bins,
                           double e_lo_deg, double e_hi_deg) {
  BlindMask m;
  m.n_ = std::max(1, n_bins);
  m.r_.assign(m.n_, std::numeric_limits<double>::infinity());
  if (h <= 1e-6) return m;

  const double two_pi = 2.0 * M_PI;
  const double az_step = M_PI / 180.0 * 0.5;   // 0.5 deg azimuth sampling
  const double el_step = M_PI / 180.0 * 0.25;  // 0.25 deg elevation sampling
  const double e_lo = e_lo_deg * M_PI / 180.0;
  const double e_hi = std::min(0.0, e_hi_deg * M_PI / 180.0);  // only downward rays hit ground

  for (double alpha = -M_PI; alpha < M_PI; alpha += az_step) {
    const double ca = std::cos(alpha), sa = std::sin(alpha);
    for (double beta = e_lo; beta <= e_hi + 1e-9; beta += el_step) {
      const double cb = std::cos(beta), sb = std::sin(beta);
      const Eigen::Vector3d d_L(cb * ca, cb * sa, sb);  // lidar-frame ray
      const Eigen::Vector3d d_B = R_base_lidar * d_L;   // base frame
      if (d_B.z() >= -1e-9) continue;                    // not pointing at the ground
      const double lambda = -h / d_B.z();                // flat-ground intersection
      const Eigen::Vector3d p_B = lambda * d_B;
      const double theta = std::atan2(p_B.y(), p_B.x());
      const double rho = std::hypot(p_B.x(), p_B.y());
      int bin = static_cast<int>(std::floor((theta + M_PI) / two_pi * m.n_));
      if (bin < 0) bin = 0;
      if (bin >= m.n_) bin = m.n_ - 1;
      if (rho < m.r_[bin]) m.r_[bin] = rho;
    }
  }
  return m;
}

double BlindMask::rBlind(double theta) const {
  if (n_ <= 0) return std::numeric_limits<double>::infinity();
  const double two_pi = 2.0 * M_PI;
  double t = std::fmod(theta + M_PI, two_pi);
  if (t < 0.0) t += two_pi;
  int bin = static_cast<int>(std::floor(t / two_pi * n_));
  if (bin < 0) bin = 0;
  if (bin >= n_) bin = n_ - 1;
  return r_[bin];
}

// ===========================================================================
// v3: first-UNKNOWN targets (bounded local scan; no synthetic points)
// ===========================================================================
std::vector<Eigen::Vector2d> computeFirstUnknownTargets(const Eigen::Vector2d& C,
                                                        const FrontierGeometry& geom,
                                                        const GridQuery& grid,
                                                        const ViewpointParams& P) {
  std::vector<Eigen::Vector2d> out;
  if (!geom.valid || !grid.isUnknown || !grid.isFree) return out;
  const double res = std::max(1e-3, grid.resolution);
  const double half_w = P.critical_frontier_half_width_m;
  const double depth = std::max(4.0 * res, 0.5);  // first UNKNOWN sits at the boundary
  const Eigen::Vector2d n = geom.normal, t = geom.tangent;

  double minx = 1e18, miny = 1e18, maxx = -1e18, maxy = -1e18;
  for (double a : {-res, depth}) {
    for (double b : {-half_w, half_w}) {
      const Eigen::Vector2d c = C + a * n + b * t;
      minx = std::min(minx, c.x()); maxx = std::max(maxx, c.x());
      miny = std::min(miny, c.y()); maxy = std::max(maxy, c.y());
    }
  }
  const int dx4[] = {1, -1, 0, 0};
  const int dy4[] = {0, 0, 1, -1};
  for (double y = miny; y <= maxy + 1e-9; y += res) {
    for (double x = minx; x <= maxx + 1e-9; x += res) {
      const Eigen::Vector2d rel(x - C.x(), y - C.y());
      if (std::abs(rel.dot(t)) > half_w) continue;  // outside the critical tangent band
      if (rel.dot(n) < -res) continue;               // free side, skip
      if (!grid.isUnknown(x, y)) continue;
      bool adj_free = false;                          // adjacent to the FREE frontier
      for (int k = 0; k < 4 && !adj_free; ++k)
        if (grid.isFree(x + dx4[k] * res, y + dy4[k] * res)) adj_free = true;
      if (adj_free) out.emplace_back(x, y);
    }
  }
  return out;
}

// ===========================================================================
// v3: local q_vis visibility maneuver
// ===========================================================================
namespace {
inline bool footprintFree(const Eigen::Vector2d& q, const GridQuery& grid,
                          const ViewpointParams& P) {
  return footprintCheck(q, grid, P) == ViewpointReject::NONE;
}
// Cheap 2-D occupancy ray: true if NO known-OCCUPIED cell blocks a->b (UNKNOWN is
// transparent -- it is exactly what we want to observe).
bool rayUnblocked(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const GridQuery& grid) {
  if (!grid.isOccupied) return true;
  const double dist = (b - a).norm();
  const int n = std::max(1, static_cast<int>(std::ceil(dist / 0.05)));
  for (int k = 1; k < n; ++k) {
    const double u = static_cast<double>(k) / n;
    const Eigen::Vector2d s = a + u * (b - a);
    if (grid.isOccupied(s.x(), s.y())) return false;
  }
  return true;
}
}  // namespace

ViewpointResult selectViewpointLocal(const Eigen::Vector2d& C, const FrontierGeometry& geom,
                                     const Eigen::Vector2d& robot_xy, const GridQuery& grid,
                                     const ViewpointParams& P, const BlindMask& mask,
                                     const std::vector<Eigen::Vector2d>& U_first,
                                     const std::vector<Eigen::Vector2d>& attempted) {
  ViewpointResult res;
  res.geom = geom;
  if (!geom.valid) { res.reason = ViewpointReject::PCA_INVALID; return res; }
  if (U_first.empty()) { res.reason = ViewpointReject::NO_TARGETS; return res; }

  Eigen::Vector2d u_bar = Eigen::Vector2d::Zero();
  for (const auto& u : U_first) u_bar += u;
  u_bar /= static_cast<double>(U_first.size());

  struct Cand {
    Eigen::Vector2d q, q_pre;
    double psi, dpsi, min_margin, clearance, robot_dist;
    int vis, tot;
  };
  std::vector<Cand> valid;

  const double step = std::max(1e-3, P.position_step_m);
  const double R = P.search_radius_m;
  const int max_pos = std::max(1, P.max_candidate_positions);
  const double n_tol = 0.5 * std::max(1e-3, grid.resolution);
  int positions_tested = 0;

  for (double dy = -R; dy <= R + 1e-9 && positions_tested < max_pos; dy += step) {
    for (double dx = -R; dx <= R + 1e-9 && positions_tested < max_pos; dx += step) {
      if (dx * dx + dy * dy > R * R) continue;
      const Eigen::Vector2d q(C.x() + dx, C.y() + dy);
      if ((q - C).dot(geom.normal) > n_tol) continue;   // stay on the known/free side of F
      if (!footprintFree(q, grid, P)) continue;
      ++positions_tested;
      bool tried = false;
      for (const auto& a : attempted)
        if ((q - a).norm() <= step) { tried = true; break; }
      if (tried) continue;

      const double psi0 = std::atan2(u_bar.y() - q.y(), u_bar.x() - q.x());
      double esdf_clear = std::numeric_limits<double>::infinity();
      if (grid.esdfInBounds && grid.esdfDistance && grid.esdfInBounds(q.x(), q.y()))
        esdf_clear = grid.esdfDistance(q.x(), q.y());

      for (double off_deg : P.heading_offsets_deg) {
        const double psi = psi0 + off_deg * M_PI / 180.0;
        const double cpsi = std::cos(psi), spsi = std::sin(psi);
        int vis = 0;
        double min_margin = std::numeric_limits<double>::infinity();
        for (const auto& u : U_first) {
          const Eigen::Vector2d vM = u - q;
          const double bx = cpsi * vM.x() + spsi * vM.y();   // rotate map->rover by -psi
          const double by = -spsi * vM.x() + cpsi * vM.y();
          const double rho = std::hypot(bx, by);
          const double theta = std::atan2(by, bx);
          if (rho - mask.rBlind(theta) < P.blind_margin_m) continue;  // in/at blind boundary
          if (!rayUnblocked(q, u, grid)) continue;                     // OCCUPIED blocks ray
          ++vis;
          min_margin = std::min(min_margin, rho - mask.rBlind(theta));
        }
        const double V_first = static_cast<double>(vis) / static_cast<double>(U_first.size());
        if (V_first < P.min_first_unknown_visible_fraction) continue;

        const Eigen::Vector2d q_pre = q - P.pre_viewpoint_len_m * Eigen::Vector2d(cpsi, spsi);
        if (!footprintFree(q_pre, grid, P)) continue;
        if (!segmentFootprintSafe(q_pre, q, grid, P)) continue;

        Cand c;
        c.q = q; c.q_pre = q_pre; c.psi = psi; c.dpsi = std::abs(off_deg);
        c.min_margin = min_margin; c.clearance = esdf_clear;
        c.robot_dist = (q_pre - robot_xy).norm(); c.vis = vis;
        c.tot = static_cast<int>(U_first.size());
        valid.push_back(c);
      }
    }
  }

  if (valid.empty()) { res.reason = ViewpointReject::LOW_VISIBILITY; return res; }

  // Lexicographic: (1) shorter robot->q_pre, (2) larger min blind margin,
  // (3) smaller |dpsi|, (4) larger ESDF clearance.
  std::sort(valid.begin(), valid.end(), [](const Cand& a, const Cand& b) {
    if (std::abs(a.robot_dist - b.robot_dist) > 1e-6) return a.robot_dist < b.robot_dist;
    if (std::abs(a.min_margin - b.min_margin) > 1e-6) return a.min_margin > b.min_margin;
    if (std::abs(a.dpsi - b.dpsi) > 1e-6) return a.dpsi < b.dpsi;
    return a.clearance > b.clearance;
  });

  const Cand& best = valid.front();
  res.ok = true;
  res.q = best.q;
  res.q_pre = best.q_pre;
  res.yaw = best.psi;
  res.s = 0.0;
  res.target = u_bar;
  res.visible_samples = best.vis;
  res.total_samples = best.tot;
  res.clearance_m = best.clearance;
  res.reason = ViewpointReject::NONE;
  return res;
}

}  // namespace mighty
