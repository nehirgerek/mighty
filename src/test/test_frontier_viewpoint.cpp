/* ----------------------------------------------------------------------------
 * Deterministic geometry tests for the frontier viewpoint selector core.
 * ROS-free: the map is a tiny analytic GridQuery, so this needs only Eigen + gtest.
 * -------------------------------------------------------------------------- */
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "mighty/frontier_viewpoint.hpp"

using namespace mighty;

namespace {

// GridQuery whose FREE/UNKNOWN split is a half-plane: UNKNOWN on the side the normal
// should point to. free_side(x,y) true => that world point is known FREE.
GridQuery halfPlaneGrid(std::function<bool(double, double)> unknown_region,
                        std::function<bool(double, double)> occ_region = nullptr,
                        double res = 0.15) {
  GridQuery g;
  g.resolution = res;
  g.isUnknown = unknown_region;
  g.isOccupied = occ_region ? occ_region : [](double, double) { return false; };
  g.isFree = [unknown_region, g_occ = g.isOccupied](double x, double y) {
    if (unknown_region && unknown_region(x, y)) return false;
    if (g_occ && g_occ(x, y)) return false;
    return true;
  };
  return g;
}

ViewpointParams baseParams() {
  ViewpointParams P;
  P.robot_bbox_x = 0.6;
  P.robot_bbox_y = 0.6;
  P.footprint_margin_m = 0.10;
  return P;
}

}  // namespace

// TEST 1 -- straight horizontal frontier: tangent ~X, normal ~Y toward UNKNOWN (y>1).
TEST(FrontierViewpoint, HorizontalFrontierGeometry) {
  std::vector<Eigen::Vector2d> cells;
  for (double x = -1.0; x <= 1.0 + 1e-9; x += 0.15) cells.emplace_back(x, 1.0);
  GridQuery grid = halfPlaneGrid([](double, double y) { return y > 1.0; });  // unknown above
  ViewpointParams P = baseParams();

  FrontierGeometry g = computeFrontierGeometry(cells, grid, P);
  ASSERT_TRUE(g.valid);
  EXPECT_GT(std::abs(g.tangent.x()), 0.9);        // tangent along X
  EXPECT_LT(std::abs(g.tangent.y()), 0.2);
  EXPECT_GT(g.normal.y(), 0.9);                   // normal points +Y (toward unknown)
}

// TEST 2 -- straight vertical frontier: tangent ~Y, normal ~X toward UNKNOWN (x>2).
TEST(FrontierViewpoint, VerticalFrontierGeometry) {
  std::vector<Eigen::Vector2d> cells;
  for (double y = -1.0; y <= 1.0 + 1e-9; y += 0.15) cells.emplace_back(2.0, y);
  GridQuery grid = halfPlaneGrid([](double x, double) { return x > 2.0; });
  ViewpointParams P = baseParams();

  FrontierGeometry g = computeFrontierGeometry(cells, grid, P);
  ASSERT_TRUE(g.valid);
  EXPECT_GT(std::abs(g.tangent.y()), 0.9);
  EXPECT_GT(g.normal.x(), 0.9);
}

// TEST 3 -- eigenvector sign is stable under a small perturbation when prev is given.
TEST(FrontierViewpoint, SignStabilityWithPrev) {
  std::vector<Eigen::Vector2d> cells;
  for (double x = -1.0; x <= 1.0 + 1e-9; x += 0.15) cells.emplace_back(x, 1.0);
  GridQuery grid = halfPlaneGrid([](double, double y) { return y > 1.0; });
  ViewpointParams P = baseParams();

  FrontierGeometry g0 = computeFrontierGeometry(cells, grid, P);
  ASSERT_TRUE(g0.valid);
  // Perturb cells slightly.
  for (auto& c : cells) c.x() += 0.01;
  FrontierGeometry g1 = computeFrontierGeometry(cells, grid, P, &g0);
  ASSERT_TRUE(g1.valid);
  EXPECT_GT(g1.tangent.dot(g0.tangent), 0.0);   // no 180-degree flip
  EXPECT_GT(g1.normal.dot(g0.normal), 0.0);
}

// TEST 4 -- compact/ambiguous cluster => PCA invalid.
TEST(FrontierViewpoint, CompactClusterInvalid) {
  // A near-square 3x3 blob has sxx ~= syy -> anisotropy ~= 1 -> no dominant direction.
  std::vector<Eigen::Vector2d> cells;
  for (double y = 0.0; y <= 0.30 + 1e-9; y += 0.15)
    for (double x = 0.0; x <= 0.30 + 1e-9; x += 0.15) cells.emplace_back(x, y);
  GridQuery grid = halfPlaneGrid([](double x, double) { return x > 0.30; });
  ViewpointParams P = baseParams();
  P.pca_min_anisotropy = 2.0;

  FrontierGeometry g = computeFrontierGeometry(cells, grid, P);
  EXPECT_FALSE(g.valid);  // blob -> low anisotropy
}

// TEST 5 -- UNKNOWN beneath the footprint disc => rejected.
TEST(FrontierViewpoint, FootprintUnknownRejected) {
  // Everything unknown -> any disc contains unknown.
  GridQuery grid = halfPlaneGrid([](double, double) { return true; });
  ViewpointParams P = baseParams();
  EXPECT_EQ(footprintCheck(Eigen::Vector2d(5.0, 5.0), grid, P),
            ViewpointReject::FOOTPRINT_UNKNOWN);
}

// TEST 6 -- OCCUPIED beneath the footprint disc => rejected (occ dominates).
TEST(FrontierViewpoint, FootprintOccupiedRejected) {
  GridQuery grid = halfPlaneGrid(
      [](double, double) { return false; },                 // nothing unknown
      [](double x, double y) { return std::hypot(x - 5.0, y - 5.0) < 0.2; });  // occ near center
  ViewpointParams P = baseParams();
  EXPECT_EQ(footprintCheck(Eigen::Vector2d(5.0, 5.0), grid, P),
            ViewpointReject::FOOTPRINT_OCCUPIED);
}

// TEST 7 -- fully known-free disc => accepted.
TEST(FrontierViewpoint, FootprintFreeAccepted) {
  GridQuery grid = halfPlaneGrid([](double, double) { return false; });  // all free
  ViewpointParams P = baseParams();
  EXPECT_EQ(footprintCheck(Eigen::Vector2d(5.0, 5.0), grid, P), ViewpointReject::NONE);
}

// TEST 8 -- lateral search returns the smallest valid |s|. s=0 sits over an unknown
// patch; s=+/-step is free -> chosen magnitude == step.
TEST(FrontierViewpoint, SmallestValidLateral) {
  // Horizontal frontier at y=0, unknown above. Viewpoint sits below at y=-standoff.
  std::vector<Eigen::Vector2d> cells;
  for (double x = -1.0; x <= 1.0 + 1e-9; x += 0.15) cells.emplace_back(x, 0.0);
  const double standoff = 0.75, step = 0.15;
  // Unknown: above the frontier, PLUS a small unknown blob exactly under q(s=0)=(0,-0.75).
  auto unknown = [&](double x, double y) {
    if (y > 0.0) return true;
    return std::hypot(x - 0.0, y - (-standoff)) < 0.05;  // tiny patch under s=0
  };
  GridQuery grid = halfPlaneGrid(unknown);
  ViewpointParams P = baseParams();
  // Small footprint so a single lateral step can clear the tiny patch (the default
  // 0.6 m box needs ~0.5 m of lateral to move the disc off a centerline obstacle).
  P.robot_bbox_x = 0.1;
  P.robot_bbox_y = 0.1;
  P.footprint_margin_m = 0.0;
  P.standoff_m = standoff;
  P.lateral_step_m = step;
  P.max_lateral_offset_m = 1.5;
  P.min_visible_fraction = 0.0;  // isolate the footprint/lateral logic

  FrontierGeometry g = computeFrontierGeometry(cells, grid, P);
  ASSERT_TRUE(g.valid);
  ViewpointResult r = selectViewpoint(Eigen::Vector2d(0.0, 0.0), g, Eigen::Vector2d(0.0, -3.0),
                                      grid, P);
  ASSERT_TRUE(r.ok);
  EXPECT_NEAR(std::abs(r.s), step, 1e-6);  // s=0 rejected, first valid magnitude is step
}

// TEST 9 -- +s / -s tie resolved by robot travel distance (robot offset toward +s).
TEST(FrontierViewpoint, TieBrokenByRobotDistance) {
  std::vector<Eigen::Vector2d> cells;
  for (double x = -1.0; x <= 1.0 + 1e-9; x += 0.15) cells.emplace_back(x, 0.0);
  const double standoff = 0.75, step = 0.15;
  auto unknown = [&](double x, double y) {
    if (y > 0.0) return true;
    return std::hypot(x, y - (-standoff)) < 0.12;  // s=0 blocked -> tie at |s|=step
  };
  GridQuery grid = halfPlaneGrid(unknown);
  ViewpointParams P = baseParams();
  P.standoff_m = standoff;
  P.lateral_step_m = step;
  P.min_visible_fraction = 0.0;

  FrontierGeometry g = computeFrontierGeometry(cells, grid, P);
  ASSERT_TRUE(g.valid);
  // Robot to the +x side => +s viewpoint is closer.
  ViewpointResult r = selectViewpoint(Eigen::Vector2d(0.0, 0.0), g, Eigen::Vector2d(2.0, -standoff),
                                      grid, P);
  ASSERT_TRUE(r.ok);
  // tangent is +/-X; +s should land on the +x side (closer to the robot).
  EXPECT_GT(r.q.x(), 0.0);
}

// TEST 10 -- target outside the vertical FOV (too steep depression) is rejected.
TEST(FrontierViewpoint, TargetOutsideVerticalFov) {
  GridQuery grid = halfPlaneGrid([](double, double) { return false; });
  ViewpointParams P = baseParams();
  // Target almost directly below: horiz tiny -> depression ~90deg -> outside FOV.
  Eigen::Vector2d q(0.0, 0.0), g(0.05, 0.0);
  EXPECT_FALSE(targetPotentiallyVisible(q, g, /*depth=*/0.5, /*standoff=*/0.75, grid, P));
  // A geometrically reasonable target passes (depth clears lip, depression in window).
  // At horiz<~1.3 m the 27deg depression limit rejects it, so use a target further out.
  Eigen::Vector2d g2(1.5, 0.0);
  EXPECT_TRUE(targetPotentiallyVisible(q, g2, /*depth=*/0.5, /*standoff=*/0.75, grid, P));
}

// TEST 11 -- target hidden by the curb lip (depth < H*d/h) is rejected.
TEST(FrontierViewpoint, TargetHiddenByLip) {
  GridQuery grid = halfPlaneGrid([](double, double) { return false; });
  ViewpointParams P = baseParams();  // H=0.15, d=0.75, h=0.51 -> lip threshold ~0.2206 m
  // Target far enough out (1.5 m) that the vertical FOV passes, so this isolates the lip:
  // depth<0.22 is lip-occluded (reject); depth>0.22 clears the lip (accept).
  Eigen::Vector2d q(0.0, 0.0), g(1.5, 0.0);
  EXPECT_FALSE(targetPotentiallyVisible(q, g, /*depth=*/0.10, /*standoff=*/0.75, grid, P));
  EXPECT_TRUE(targetPotentiallyVisible(q, g, /*depth=*/0.40, /*standoff=*/0.75, grid, P));
}

// TEST 13 -- pre-viewpoint q_pre sits behind q toward the target so the final leg
// q_pre->q points at the unknown; the leg is footprint-safe.
TEST(FrontierViewpoint, PreViewpointBehindTowardTarget) {
  std::vector<Eigen::Vector2d> cells;
  for (double x = -1.0; x <= 1.0 + 1e-9; x += 0.15) cells.emplace_back(x, 0.0);
  GridQuery grid = halfPlaneGrid([](double, double y) { return y > 0.0; });  // unknown above
  ViewpointParams P = baseParams();
  P.standoff_m = 0.75;
  P.pre_viewpoint_len_m = 0.50;
  P.min_visible_fraction = 0.0;

  FrontierGeometry g = computeFrontierGeometry(cells, grid, P);
  ASSERT_TRUE(g.valid);
  ViewpointResult r = selectViewpoint(Eigen::Vector2d(0.0, 0.0), g, Eigen::Vector2d(0.0, -3.0),
                                      grid, P);
  ASSERT_TRUE(r.ok);
  // q is below the frontier (-n), target is above (+n): q_pre is further below q, and
  // the approach direction q_pre->q points toward the target (+y).
  EXPECT_LT(r.q_pre.y(), r.q.y() - 0.4);
  const Eigen::Vector2d approach = (r.q - r.q_pre).normalized();
  EXPECT_GT(approach.y(), 0.9);
}

// TEST 14 -- if the pre-viewpoint approach segment leaves known-free space, the
// candidate is rejected APPROACH_UNSAFE (not silently accepted).
TEST(FrontierViewpoint, ApproachUnsafeRejected) {
  std::vector<Eigen::Vector2d> cells;
  for (double x = -1.0; x <= 1.0 + 1e-9; x += 0.15) cells.emplace_back(x, 0.0);
  // Free only for -1.5 <= y <= 0; unknown above the frontier and far below (where q_pre lands).
  GridQuery grid = halfPlaneGrid([](double, double y) { return y > 0.0 || y < -1.5; });
  ViewpointParams P = baseParams();
  P.standoff_m = 0.75;
  P.pre_viewpoint_len_m = 1.0;  // pushes q_pre (~y=-1.75) into the unknown band
  P.min_visible_fraction = 0.0;

  FrontierGeometry g = computeFrontierGeometry(cells, grid, P);
  ASSERT_TRUE(g.valid);
  ViewpointResult r = selectViewpoint(Eigen::Vector2d(0.0, 0.0), g, Eigen::Vector2d(0.0, -1.0),
                                      grid, P);
  EXPECT_FALSE(r.ok);
  EXPECT_EQ(r.reason, ViewpointReject::APPROACH_UNSAFE);
}

// TEST 15 -- PCA-aligned strip snapshot collects UNKNOWN cells BEHIND the frontier
// only, and revealFraction reflects how many later became known.
TEST(FrontierViewpoint, StripSnapshotAndReveal) {
  std::vector<Eigen::Vector2d> cells;
  for (double x = -1.0; x <= 1.0 + 1e-9; x += 0.15) cells.emplace_back(x, 0.0);
  // Snapshot-time grid: unknown above the frontier (y>0).
  bool revealed = false;
  auto unknown_fn = [&revealed](double, double y) {
    if (revealed) return false;  // after "observation", nothing unknown remains in strip
    return y > 0.0;
  };
  GridQuery grid = halfPlaneGrid(unknown_fn);
  ViewpointParams P = baseParams();
  P.strip_half_width_m = 0.75;
  P.strip_depth_m = 0.70;

  FrontierGeometry g = computeFrontierGeometry(cells, grid, P);
  ASSERT_TRUE(g.valid);
  auto snap = snapshotTargetStripUnknown(Eigen::Vector2d(0.0, 0.0), g, grid, P);
  ASSERT_GT(snap.size(), 0u);
  for (const auto& c : snap) {  // all snapshot cells are behind the frontier (+normal, y>0)
    EXPECT_GT(c.y(), -1e-9);
  }
  EXPECT_NEAR(revealFraction(snap, grid), 0.0, 1e-9);  // nothing revealed yet
  revealed = true;
  EXPECT_NEAR(revealFraction(snap, grid), 1.0, 1e-9);  // all cells now known
}

// TEST 16 -- ranked alternates: more than one valid viewpoint is returned, best-first
// by |s|, so the node can fall back if the reveal test fails.
TEST(FrontierViewpoint, RankedAlternatesProvided) {
  std::vector<Eigen::Vector2d> cells;
  for (double x = -1.0; x <= 1.0 + 1e-9; x += 0.15) cells.emplace_back(x, 0.0);
  GridQuery grid = halfPlaneGrid([](double, double y) { return y > 0.0; });  // open free below
  ViewpointParams P = baseParams();
  P.standoff_m = 0.75;
  P.min_visible_fraction = 0.0;
  P.max_lateral_offset_m = 0.60;

  FrontierGeometry g = computeFrontierGeometry(cells, grid, P);
  ASSERT_TRUE(g.valid);
  ViewpointResult r = selectViewpoint(Eigen::Vector2d(0.0, 0.0), g, Eigen::Vector2d(0.0, -3.0),
                                      grid, P);
  ASSERT_TRUE(r.ok);
  ASSERT_GT(r.ranked.size(), 1u);
  EXPECT_NEAR(std::abs(r.ranked[0].s), 0.0, 1e-9);                       // best is s=0
  EXPECT_GE(std::abs(r.ranked[1].s), std::abs(r.ranked[0].s) - 1e-9);   // then larger |s|
}

// TEST 17 -- a TF extrinsic of Ry(20deg) reproduces the scalar sensor_mount_pitch_deg=20
// FOV classification exactly (the TF path and the fallback agree).
TEST(FrontierViewpoint, TfExtrinsicMatchesScalarPitch) {
  GridQuery grid = halfPlaneGrid([](double, double) { return false; });
  ViewpointParams P = baseParams();
  P.sensor_mount_pitch_deg = 20.0;  // scalar fallback
  SensorExtrinsics extr;
  extr.valid = true;
  extr.R_base_lidar = Eigen::AngleAxisd(20.0 * M_PI / 180.0, Eigen::Vector3d::UnitY()).toRotationMatrix();

  const Eigen::Vector2d q(0.0, 0.0);
  for (double horiz : {0.5, 0.9, 1.2, 1.3, 1.5, 2.0, 3.0}) {
    const Eigen::Vector2d g(horiz, 0.0);
    const bool scalar = targetPotentiallyVisible(q, g, 0.5, 0.75, grid, P);            // fallback (invalid extr)
    const bool tf     = targetPotentiallyVisible(q, g, 0.5, 0.75, grid, P, 0.0, extr); // TF Ry(20)
    EXPECT_EQ(scalar, tf) << "horiz=" << horiz;
  }
}

// TEST 18 -- FOV classification responds to the sensor extrinsic orientation.
TEST(FrontierViewpoint, FovRespondsToOrientation) {
  GridQuery grid = halfPlaneGrid([](double, double) { return false; });
  ViewpointParams P = baseParams();
  const Eigen::Vector2d q(0.0, 0.0), g(2.0, 0.0);  // 2 m out -> ~18 deg depression
  SensorExtrinsics flat;    flat.valid = true;     flat.R_base_lidar = Eigen::Matrix3d::Identity();
  SensorExtrinsics steep;   steep.valid = true;
  steep.R_base_lidar = Eigen::AngleAxisd(40.0 * M_PI / 180.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
  // Flat sensor: target is below the -7deg lower bound -> not visible.
  EXPECT_FALSE(targetPotentiallyVisible(q, g, 0.5, 0.75, grid, P, 0.0, flat));
  // Pitched 40deg down: the same target rises into the vertical FOV -> visible.
  EXPECT_TRUE(targetPotentiallyVisible(q, g, 0.5, 0.75, grid, P, 0.0, steep));
}

// TEST 19 -- an empty target strip yields reveal 0.0 (never a successful observation).
TEST(FrontierViewpoint, EmptyStripRevealIsZero) {
  GridQuery all_free = halfPlaneGrid([](double, double) { return false; });  // nothing unknown
  std::vector<Eigen::Vector2d> cells;
  for (double x = -1.0; x <= 1.0 + 1e-9; x += 0.15) cells.emplace_back(x, 0.0);
  ViewpointParams P = baseParams();
  P.strip_depth_m = 0.70;
  FrontierGeometry g = computeFrontierGeometry(cells, all_free, P);
  ASSERT_TRUE(g.valid);
  auto snap = snapshotTargetStripUnknown(Eigen::Vector2d(0.0, 0.0), g, all_free, P);
  EXPECT_TRUE(snap.empty());
  EXPECT_DOUBLE_EQ(revealFraction(snap, all_free), 0.0);  // empty != fully revealed
  EXPECT_DOUBLE_EQ(revealFraction({}, all_free), 0.0);
}

// TEST 20 -- reveal threshold: R=0.29 fails, R=0.30 succeeds (min_reveal_fraction=0.30).
TEST(FrontierViewpoint, RevealThreshold) {
  std::vector<Eigen::Vector2d> snap;
  for (int i = 0; i < 100; ++i) snap.emplace_back(static_cast<double>(i), 0.0);
  int revealed = 0;
  GridQuery grid;
  grid.isUnknown = [&revealed](double x, double) { return x >= static_cast<double>(revealed); };
  const double thresh = 0.30;
  revealed = 29;  // 71 still unknown -> R = 0.29
  EXPECT_LT(revealFraction(snap, grid), thresh);
  revealed = 30;  // 70 still unknown -> R = 0.30
  EXPECT_GE(revealFraction(snap, grid), thresh);
}

// ---- v3: blind mask + local q_vis selector -------------------------------
namespace {
Eigen::Matrix3d Rpitch(double deg) {
  return Eigen::AngleAxisd(deg * M_PI / 180.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
}
// Frontier along x=0 (tangent ~y), UNKNOWN at x>0; optional occupied band.
GridQuery frontierGrid(std::function<bool(double, double)> occ = nullptr) {
  return halfPlaneGrid([](double x, double) { return x > 0.0; }, occ);
}
std::vector<Eigen::Vector2d> frontierCells() {
  std::vector<Eigen::Vector2d> c;
  for (double y = -0.6; y <= 0.6 + 1e-9; y += 0.15) c.emplace_back(0.0, y);
  return c;
}
}  // namespace

// TEST 21 -- blind mask: 20deg pitch gives a plausible forward near-ground range, and a
// second identical build (the scalar fallback path uses the same Ry) matches bin-for-bin.
TEST(FrontierViewpoint, BlindMaskScalarVsTfPitch) {
  BlindMask a = BlindMask::build(Rpitch(20.0), 0.51, 360, -7.0, 52.0);
  BlindMask b = BlindMask::build(Rpitch(20.0), 0.51, 360, -7.0, 52.0);
  ASSERT_TRUE(a.valid());
  const double fwd = a.rBlind(0.0);
  EXPECT_GT(fwd, 0.7);   // ~ h/tan(pitch - vmin) = 0.51/tan(27deg) ~ 1.0 m
  EXPECT_LT(fwd, 1.4);
  for (double th = -M_PI; th < M_PI; th += 0.1) EXPECT_DOUBLE_EQ(a.rBlind(th), b.rBlind(th));
}

// TEST 22 -- pitched LiDAR makes r_blind heading-dependent: forward sees closer than back.
TEST(FrontierViewpoint, BlindMaskIsHeadingDependent) {
  BlindMask m = BlindMask::build(Rpitch(20.0), 0.51, 360, -7.0, 52.0);
  ASSERT_TRUE(m.valid());
  EXPECT_LT(m.rBlind(0.0), m.rBlind(M_PI));  // forward blind boundary is nearer than rear
}

// TEST 23 -- no fixed standoff: only a candidate far enough to clear the forward blind
// zone works, so the selector picks q_vis well beyond the old 0.75 m standoff.
TEST(FrontierViewpoint, LocalSelectorNoFixedStandoff) {
  GridQuery grid = frontierGrid();
  ViewpointParams P = baseParams();
  P.min_first_unknown_visible_fraction = 1.0;
  FrontierGeometry g = computeFrontierGeometry(frontierCells(), grid, P);
  ASSERT_TRUE(g.valid);
  EXPECT_GT(g.normal.x(), 0.9);  // +x toward UNKNOWN
  BlindMask mask = BlindMask::build(Rpitch(20.0), P.sensor_height_m, 360,
                                    P.sensor_vertical_min_deg, P.sensor_vertical_max_deg);
  auto U = computeFirstUnknownTargets(Eigen::Vector2d(0, 0), g, grid, P);
  ASSERT_GT(U.size(), 0u);

  ViewpointResult r = selectViewpointLocal(Eigen::Vector2d(0, 0), g, Eigen::Vector2d(-2.0, 0.0),
                                           grid, P, mask, U);
  ASSERT_TRUE(r.ok);
  EXPECT_GT(-r.q.x(), 1.0);  // chose a pose > 1 m back (not the 0.75 m standoff)

  // With a small search disk no far-enough pose exists -> no viewpoint.
  ViewpointParams P2 = P;
  P2.search_radius_m = 0.8;
  ViewpointResult r2 = selectViewpointLocal(Eigen::Vector2d(0, 0), g, Eigen::Vector2d(-2.0, 0.0),
                                            grid, P2, mask, U);
  EXPECT_FALSE(r2.ok);
}

// TEST 24 -- a known OCCUPIED wall between q_vis and the first UNKNOWN blocks visibility.
TEST(FrontierViewpoint, LocalSelectorWallOccludes) {
  GridQuery grid = frontierGrid([](double x, double) { return x > -0.4 && x < -0.3; });  // wall
  ViewpointParams P = baseParams();
  P.min_first_unknown_visible_fraction = 1.0;
  FrontierGeometry g = computeFrontierGeometry(frontierCells(), grid, P);
  ASSERT_TRUE(g.valid);
  BlindMask mask = BlindMask::build(Rpitch(20.0), P.sensor_height_m, 360,
                                    P.sensor_vertical_min_deg, P.sensor_vertical_max_deg);
  auto U = computeFirstUnknownTargets(Eigen::Vector2d(0, 0), g, grid, P);
  ASSERT_GT(U.size(), 0u);
  ViewpointResult r = selectViewpointLocal(Eigen::Vector2d(0, 0), g, Eigen::Vector2d(-2.0, 0.0),
                                           grid, P, mask, U);
  EXPECT_FALSE(r.ok);  // every far pose's ray crosses the occupied wall
}

// TEST 25 -- q_pre is derived so the q_pre->q_vis heading equals the selected yaw.
TEST(FrontierViewpoint, LocalSelectorQPreDerivation) {
  GridQuery grid = frontierGrid();
  ViewpointParams P = baseParams();
  P.min_first_unknown_visible_fraction = 1.0;
  FrontierGeometry g = computeFrontierGeometry(frontierCells(), grid, P);
  ASSERT_TRUE(g.valid);
  BlindMask mask = BlindMask::build(Rpitch(20.0), P.sensor_height_m, 360,
                                    P.sensor_vertical_min_deg, P.sensor_vertical_max_deg);
  auto U = computeFirstUnknownTargets(Eigen::Vector2d(0, 0), g, grid, P);
  ViewpointResult r = selectViewpointLocal(Eigen::Vector2d(0, 0), g, Eigen::Vector2d(-2.0, 0.0),
                                           grid, P, mask, U);
  ASSERT_TRUE(r.ok);
  const double seg_yaw = std::atan2(r.q.y() - r.q_pre.y(), r.q.x() - r.q_pre.x());
  const double err = std::atan2(std::sin(seg_yaw - r.yaw), std::cos(seg_yaw - r.yaw));
  EXPECT_NEAR(err, 0.0, 1e-6);
}

// ---- Step 9: pre-observation route known-FREE validation ------------------
namespace {
// A grid FREE inside [0,10]x[0,10] (minus unknown/occ regions); everything else OOB
// (=> UNKNOWN semantics). unk/occ are optional analytic regions in-bounds.
GridQuery boundedGrid(std::function<bool(double, double)> unk = nullptr,
                      std::function<bool(double, double)> occ = nullptr) {
  GridQuery g;
  g.resolution = 0.15;
  g.isOccupied = [occ](double x, double y) { return occ && occ(x, y); };
  g.isUnknown = [unk](double x, double y) {
    if (x < 0.0 || x > 10.0 || y < 0.0 || y > 10.0) return true;  // OOB => unknown
    return unk && unk(x, y);
  };
  g.isFree = [g](double x, double y) {
    if (x < 0.0 || x > 10.0 || y < 0.0 || y > 10.0) return false;      // OOB not free
    return !g.isUnknown(x, y) && !g.isOccupied(x, y);
  };
  return g;
}
ViewpointParams pathParams() {
  ViewpointParams P;
  P.robot_bbox_x = 0.6; P.robot_bbox_y = 0.6; P.footprint_margin_m = 0.10;  // r_safe ~0.52
  return P;
}
}  // namespace

// 1 - entirely known-FREE -> pass.
TEST(FrontierViewpoint, PrePathAllFree) {
  GridQuery g = boundedGrid();
  ViewpointParams P = pathParams();
  std::vector<Eigen::Vector2d> path = {{2, 5}, {3, 5}, {4, 5}};
  EXPECT_EQ(pathFootprintKnownFree(path, g, P), ViewpointReject::NONE);
}
// 2 - centerline enters UNKNOWN -> fail.
TEST(FrontierViewpoint, PrePathCenterlineUnknown) {
  GridQuery g = boundedGrid([](double x, double) { return x > 4.0; });  // unknown x>4
  ViewpointParams P = pathParams();
  std::vector<Eigen::Vector2d> path = {{2, 5}, {5, 5}};  // ends in unknown
  EXPECT_EQ(pathFootprintKnownFree(path, g, P), ViewpointReject::FOOTPRINT_UNKNOWN);
}
// 3 - footprint overlaps UNKNOWN while centerline is FREE -> fail.
TEST(FrontierViewpoint, PrePathFootprintUnknown) {
  GridQuery g = boundedGrid([](double, double y) { return y > 5.3; });  // unknown y>5.3
  ViewpointParams P = pathParams();  // footprint disc at y=5.0 reaches into y>5.3
  std::vector<Eigen::Vector2d> path = {{2, 5.0}, {4, 5.0}};  // centerline free, disc clips unknown
  EXPECT_EQ(pathFootprintKnownFree(path, g, P), ViewpointReject::FOOTPRINT_UNKNOWN);
}
// 4 - footprint overlaps OCCUPIED -> fail (occ dominates).
TEST(FrontierViewpoint, PrePathFootprintOccupied) {
  GridQuery g = boundedGrid(nullptr, [](double x, double y) {
    return std::hypot(x - 3.3, y - 5.0) < 0.2;  // small occupied blob near the path
  });
  ViewpointParams P = pathParams();
  std::vector<Eigen::Vector2d> path = {{2, 5.0}, {4, 5.0}};
  EXPECT_EQ(pathFootprintKnownFree(path, g, P), ViewpointReject::FOOTPRINT_OCCUPIED);
}
// 5 - UNKNOWN between two sparse vertices -> fail (interpolation catches it).
TEST(FrontierViewpoint, PrePathUnknownBetweenSparseVertices) {
  GridQuery g = boundedGrid([](double x, double) { return x > 4.5 && x < 5.5; });  // unknown slab
  ViewpointParams P = pathParams();
  std::vector<Eigen::Vector2d> path = {{2, 5}, {8, 5}};  // sparse: slab is only mid-segment
  EXPECT_EQ(pathFootprintKnownFree(path, g, P), ViewpointReject::FOOTPRINT_UNKNOWN);
}
// 6 - OOB footprint -> fail.
TEST(FrontierViewpoint, PrePathOutOfBounds) {
  GridQuery g = boundedGrid();
  ViewpointParams P = pathParams();
  std::vector<Eigen::Vector2d> path = {{2, 5}, {0.2, 5}};  // disc reaches x<0 (OOB)
  EXPECT_EQ(pathFootprintKnownFree(path, g, P), ViewpointReject::FOOTPRINT_UNKNOWN);
}
// 7 - first candidate route fails, second succeeds (node's "try next ranked" proxy).
TEST(FrontierViewpoint, PrePathFirstFailsSecondSucceeds) {
  GridQuery g = boundedGrid([](double x, double) { return x > 4.0 && x < 4.6; });  // unknown wall
  ViewpointParams P = pathParams();
  std::vector<Eigen::Vector2d> cand1 = {{2, 5}, {6, 5}};        // crosses the unknown wall
  std::vector<Eigen::Vector2d> cand2 = {{2, 5}, {2, 8}, {2, 2}};  // detour stays free
  EXPECT_NE(pathFootprintKnownFree(cand1, g, P), ViewpointReject::NONE);
  EXPECT_EQ(pathFootprintKnownFree(cand2, g, P), ViewpointReject::NONE);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
