// /* ----------------------------------------------------------------------------
//  * Deterministic tests for the perception-aware lattice A* cost integration.
//  *
//  * These exercise the standalone hgp::planPerceptionAware() with an injected
//  * hgp::PerceptionCost (heat map + cutoff), so they need neither map_util nor a ROS
//  * node -- the same decoupling the MIGHTY integration relies on.
//  *
//  * Covers (see task spec):
//  *   1 free map            -> reaches goal, no heat/unknown cost
//  *   2 two equal corridors -> picks the lower-heat one
//  *   3 short-hot vs long-cool -> heat_weight flips the route
//  *   4 hard heat cutoff    -> never traverses a cell above the cutoff
//  *   5 unknown cost        -> higher w_unknown avoids unknown cells
//  *   6 perception is HARD  -> never takes a low-cost but coverage-infeasible path
//  *   (7 = "perception disabled leaves GraphSearch astar_heat untouched": guaranteed by
//  *    NOT modifying getSucc / the non-perception plan() path; nothing to unit-test
//  *    against GraphSearch without a full map_util, so it is asserted by inspection.)
//  * -------------------------------------------------------------------------- */
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "hgp/perception_planner.hpp"
#include "mighty/occ_grid_2d.hpp"

using hgp::AnnulusSensor;
using hgp::PerceptionCost;
using hgp::PerceptionParams;
using hgp::PerceptionPlanResult;

namespace {

constexpr int8_t FREE = 0;
constexpr int8_t OCC = 100;
constexpr int8_t UNK = -1;

// Simple test world with a co-located heat field (same grid as the belief).
struct World {
  int nx, ny;
  double res, ox, oy;
  std::vector<int8_t> cells;   // FREE / OCC / UNK
  std::vector<float> heat;     // per-cell heat value
  std::shared_ptr<const OccGrid2D> belief;

  World(int nx_, int ny_, double res_, double ox_ = 0.0, double oy_ = 0.0)
      : nx(nx_), ny(ny_), res(res_), ox(ox_), oy(oy_),
        cells(static_cast<size_t>(nx_) * ny_, FREE),
        heat(static_cast<size_t>(nx_) * ny_, 0.0f) {}

  void setRect(int x0, int y0, int x1, int y1, int8_t v) {
    for (int y = y0; y < y1; ++y)
      for (int x = x0; x < x1; ++x) cells[y * nx + x] = v;
  }
  void heatRect(int x0, int y0, int x1, int y1, float v) {
    for (int y = y0; y < y1; ++y)
      for (int x = x0; x < x1; ++x) heat[y * nx + x] = v;
  }
  void build() { belief = OccGrid2D::fromTristate(nx, ny, res, ox, oy, cells); }

  void cellOf(double wx, double wy, int& cx, int& cy) const {
    cx = static_cast<int>(std::floor((wx - ox) / res));
    cy = static_cast<int>(std::floor((wy - oy) / res));
  }
  float heatAt(double wx, double wy) const {
    int cx, cy;
    cellOf(wx, wy, cx, cy);
    if (cx < 0 || cx >= nx || cy < 0 || cy >= ny) return 0.0f;
    return heat[cy * nx + cx];
  }
  bool unknownAt(double wx, double wy) const {
    int cx, cy;
    cellOf(wx, wy, cx, cy);
    if (cx < 0 || cx >= nx || cy < 0 || cy >= ny) return true;
    return cells[cy * nx + cx] == UNK;
  }
};

PerceptionParams baseParams(double res) {
  PerceptionParams P;
  P.res = res;
  P.n_headings = 12;
  P.prim_len = 1.0;
  P.robot_radius = 0.0;   // no inflation: keep test geometry exact
  P.turn_cost = 0.35;
  P.w_unknown = 0.0;
  P.goal_tol = 0.30;
  P.use_coverage_rule = false;  // most cost tests isolate cost from the coverage rule
  P.back_projection_steps = 5;
  P.back_projection_step = 0.25;
  P.max_expand = 500000;
  P.timeout_ms = 0;  // no wall-clock cap in tests (determinism)
  return P;
}

// A heat cost hook: cellPenalty = w_heat * heat; hardBlocked = heat > cutoff.
PerceptionCost heatCost(const World& w, double w_heat, double cutoff = -1.0) {
  PerceptionCost c;
  c.cellPenalty = [&w, w_heat](double wx, double wy) -> double {
    return w_heat * static_cast<double>(w.heatAt(wx, wy));
  };
  if (cutoff >= 0.0) {
    c.hardBlocked = [&w, cutoff](double wx, double wy) -> bool {
      return static_cast<double>(w.heatAt(wx, wy)) > cutoff;
    };
  }
  return c;
}

// Sum raw (unweighted) heat exposure over the returned path's pose cells.
double rawHeatOnPath(const World& w, const PerceptionPlanResult& r) {
  double s = 0.0;
  for (const auto& p : r.states) s += w.heatAt(p[0], p[1]);
  return s;
}
int unknownCountOnPath(const World& w, const PerceptionPlanResult& r) {
  int n = 0;
  for (const auto& p : r.states)
    if (w.unknownAt(p[0], p[1])) ++n;
  return n;
}
double residualToGoal(const PerceptionPlanResult& r, double gx, double gy) {
  if (r.states.empty()) return 1e9;
  const auto& b = r.states.back();
  return std::hypot(gx - b[0], gy - b[1]);
}

}  // namespace

// TEST 1 -- free map, no heat, no unknown: reaches the goal at ~motion-only cost.
TEST(PerceptionCost, FreeMapShortestRoute) {
  World w(50, 25, 0.2);  // 10m x 5m
  w.build();
  PerceptionParams P = baseParams(w.res);
  AnnulusSensor sensor;
  const double sx = 1.0, sy = 2.5, gx = 8.0, gy = 2.5;

  PerceptionPlanResult r =
      hgp::planPerceptionAware(*w.belief, sensor, P, sx, sy, 0.0, gx, gy);

  ASSERT_TRUE(r.ok);
  EXPECT_LT(residualToGoal(r, gx, gy), P.goal_tol + w.res);
  EXPECT_DOUBLE_EQ(r.heat_cost, 0.0);
  EXPECT_DOUBLE_EQ(r.unknown_cost, 0.0);
  // Motion within a reasonable factor of the straight-line distance (lattice, so not exact).
  EXPECT_LT(r.motion_cost, 1.4 * (gx - sx));
}

// TEST 2 -- two equal-length corridors, one hot: planner takes the cool one.
TEST(PerceptionCost, PrefersLowerHeatCorridor) {
  World w(50, 25, 0.2);
  // Wall across the middle rows leaves a top corridor (y high) and a bottom corridor
  // (y low), both open along x. Put heat only in the TOP corridor.
  w.setRect(5, 11, 45, 14, OCC);   // dividing wall (rows 11..13)
  w.heatRect(5, 14, 45, 25, 5.0f);  // top corridor hot
  w.build();

  PerceptionParams P = baseParams(w.res);
  AnnulusSensor sensor;
  const double sx = 1.0, sy = 2.5, gx = 8.0, gy = 2.5;  // start/goal on the bottom side
  PerceptionCost c = heatCost(w, /*w_heat=*/2.0);

  PerceptionPlanResult r =
      hgp::planPerceptionAware(*w.belief, sensor, P, sx, sy, 0.0, gx, gy, nullptr, &c);

  ASSERT_TRUE(r.ok);
  EXPECT_LT(residualToGoal(r, gx, gy), P.goal_tol + w.res);
  // The bottom (cool) corridor connects start->goal directly; the path should not climb
  // into the hot top corridor.
  EXPECT_NEAR(rawHeatOnPath(w, r), 0.0, 1e-9);
}

// TEST 3 -- heat_weight changes the route (short-hot vs slightly-longer-cool).
TEST(PerceptionCost, HeatWeightFlipsRoute) {
  World w(60, 30, 0.2);
  // A hot band straddling the direct line y=3.0; cool space above/below for a detour.
  w.heatRect(15, 13, 45, 17, 8.0f);  // rows 13..16 hot across the middle
  w.build();

  PerceptionParams P = baseParams(w.res);
  AnnulusSensor sensor;
  const double sx = 1.0, sy = 3.0, gx = 10.0, gy = 3.0;

  PerceptionCost cool = heatCost(w, /*w_heat=*/0.0);   // ignore heat -> straight through
  PerceptionCost hot = heatCost(w, /*w_heat=*/5.0);    // penalize heat -> detour

  PerceptionPlanResult r_lo =
      hgp::planPerceptionAware(*w.belief, sensor, P, sx, sy, 0.0, gx, gy, nullptr, &cool);
  PerceptionPlanResult r_hi =
      hgp::planPerceptionAware(*w.belief, sensor, P, sx, sy, 0.0, gx, gy, nullptr, &hot);

  ASSERT_TRUE(r_lo.ok);
  ASSERT_TRUE(r_hi.ok);
  // Penalizing heat should reduce heat exposure and (weakly) lengthen the motion.
  EXPECT_LE(rawHeatOnPath(w, r_hi), rawHeatOnPath(w, r_lo));
  EXPECT_LT(rawHeatOnPath(w, r_hi), rawHeatOnPath(w, r_lo));  // strictly avoids some heat
  EXPECT_GE(r_hi.motion_cost, r_lo.motion_cost - 1e-9);
}

// TEST 4 -- hard heat cutoff: a cell above the cutoff is impassable, never traversed.
TEST(PerceptionCost, HardHeatCutoffRejectsCell) {
  World w(50, 25, 0.2);
  // Full-height hot wall across a column band, heat above the cutoff -> blocks the route.
  w.heatRect(20, 0, 24, 25, 100.0f);
  w.build();

  PerceptionParams P = baseParams(w.res);
  AnnulusSensor sensor;
  const double sx = 1.0, sy = 2.5, gx = 8.0, gy = 2.5;
  PerceptionCost c = heatCost(w, /*w_heat=*/1.0, /*cutoff=*/50.0);

  PerceptionPlanResult r =
      hgp::planPerceptionAware(*w.belief, sensor, P, sx, sy, 0.0, gx, gy, nullptr, &c);

  // The planner may return a partial path, but it must NEVER stand on an above-cutoff
  // cell, and it must not have reached the goal beyond the wall.
  for (const auto& p : r.states) EXPECT_LE(w.heatAt(p[0], p[1]), 50.0f);
  EXPECT_GT(residualToGoal(r, gx, gy), P.goal_tol);  // goal is unreachable through the wall
}

// TEST 5 -- unknown cost: raising w_unknown steers away from unknown cells.
TEST(PerceptionCost, UnknownCostAvoidsUnknown) {
  World w(60, 30, 0.2);
  // Unknown band straddling the direct line; free space above/below for a detour.
  w.setRect(15, 13, 45, 17, UNK);
  w.build();

  AnnulusSensor sensor;
  const double sx = 1.0, sy = 3.0, gx = 10.0, gy = 3.0;

  PerceptionParams P_lo = baseParams(w.res);
  P_lo.w_unknown = 0.0;  // indifferent to unknown -> straight through
  PerceptionParams P_hi = baseParams(w.res);
  P_hi.w_unknown = 5.0;  // strong unknown penalty -> detour around

  PerceptionPlanResult r_lo =
      hgp::planPerceptionAware(*w.belief, sensor, P_lo, sx, sy, 0.0, gx, gy);
  PerceptionPlanResult r_hi =
      hgp::planPerceptionAware(*w.belief, sensor, P_hi, sx, sy, 0.0, gx, gy);

  ASSERT_TRUE(r_lo.ok);
  ASSERT_TRUE(r_hi.ok);
  EXPECT_LT(unknownCountOnPath(w, r_hi), unknownCountOnPath(w, r_lo));
  EXPECT_GE(r_hi.motion_cost, r_lo.motion_cost - 1e-9);
}

// TEST 6 -- perception is a HARD constraint, not a penalty. A low-cost straight path
// through unobservable unknown must be rejected in favor of a feasible detour.
TEST(PerceptionCost, PerceptionRuleIsHardNotWeighted) {
  World w(60, 30, 0.2);
  // Unknown band across the direct line; free detour above/below.
  w.setRect(15, 13, 45, 17, UNK);
  w.build();

  // Sensor with a 0.5 m blind annulus, and NO back-projection ladder, so unknown cells
  // on the approach line cannot be certified -> the straight route is infeasible.
  AnnulusSensor sensor(/*r_min=*/0.5, /*r_max=*/3.0, /*fov_deg=*/360.0);
  const double sx = 1.0, sy = 3.0, gx = 10.0, gy = 3.0;

  PerceptionParams P = baseParams(w.res);
  P.use_coverage_rule = true;
  P.back_projection_steps = 0;  // disarm the ladder -> blind-annulus unknown is infeasible

  // Control: coverage OFF -> takes the cheap straight route through the unknown band.
  PerceptionParams P_off = P;
  P_off.use_coverage_rule = false;

  PerceptionPlanResult r_on =
      hgp::planPerceptionAware(*w.belief, sensor, P, sx, sy, 0.0, gx, gy);
  PerceptionPlanResult r_off =
      hgp::planPerceptionAware(*w.belief, sensor, P_off, sx, sy, 0.0, gx, gy);

  ASSERT_TRUE(r_on.ok);
  ASSERT_TRUE(r_off.ok);
  // Coverage ON must NOT drive through the unobservable unknown band...
  EXPECT_EQ(unknownCountOnPath(w, r_on), 0);
  // ...while the cheaper control route does.
  EXPECT_GT(unknownCountOnPath(w, r_off), 0);
  // And the feasible route is at least as long as the rejected cheap one.
  EXPECT_GE(r_on.motion_cost, r_off.motion_cost - 1e-9);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
