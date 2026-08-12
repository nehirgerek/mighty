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
                              double standoff, const GridQuery& grid, const ViewpointParams& P) {
  const double h = P.sensor_height_m;
  const double H = P.critical_drop_m;
  const double d = standoff;

  // (a) Curb-lip clearance: a lower-ground target at depth x behind a straight lip of
  // height H, seen from perpendicular distance d with sensor height h, is occluded
  // until x >= H*d/h. The lateral offset does not remove straight-lip occlusion.
  if (h > 1e-6 && depth < (H * d / h)) return false;

  // (b) Mid-360 vertical-FOV depression window. Sensor faces the target; horizontal
  // range is the true (possibly diagonal) distance, vertical drop is h + H (target on
  // lower ground). Depression delta is positive downward. A nose-down mount pitch p
  // maps a world ray of elevation e to sensor elevation (e + p); here e = -delta, so
  // the sensor elevation is (p - delta) and must lie within the native [vmin, vmax].
  const double horiz = (g - q).norm();
  const double delta = std::atan2(h + H, std::max(horiz, 1e-6));  // downward, +
  const double p = P.sensor_mount_pitch_deg * kDeg2Rad;
  const double sensor_elev = p - delta;
  const double vmin = P.sensor_vertical_min_deg * kDeg2Rad;
  const double vmax = P.sensor_vertical_max_deg * kDeg2Rad;
  if (sensor_elev < vmin || sensor_elev > vmax) return false;

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
  if (snapshot_unknown.empty()) return 1.0;   // nothing was unknown -> treat as fully revealed
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
                              const ViewpointParams& P, Eigen::Vector2d& q_out,
                              Eigen::Vector2d& q_pre_out, int& vis_out, int& tot_out,
                              double& clear_out, std::vector<Eigen::Vector2d>& samples_out) {
  const Eigen::Vector2d q = C - P.standoff_m * geom.normal + s * geom.tangent;
  q_out = q;
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
    if (targetPotentiallyVisible(q, g, depth, P.standoff_m, grid, P)) {
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
                                const ViewpointParams& P) {
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
      const ViewpointReject r =
          evalCandidate(offs[i], centroid, geom, target, grid, P, q, q_pre, vis, tot, clr, samples);
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

}  // namespace mighty
