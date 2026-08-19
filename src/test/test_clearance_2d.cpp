// Tests for the 2D endpoint-clearance field and goal projection free functions
// (mighty::computeClearanceField2D / mighty::projectToClearance2D). These are
// deliberately free functions over raw (width, height, res, map/clearance)
// views, so they exercise the wavefront and BFS WITHOUT constructing a MapUtil
// (which drags in PCL / Eigen / decomp). See include/hgp/clearance_2d.hpp.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "hgp/clearance_2d.hpp"

using mighty::computeClearanceField2D;
using mighty::projectToClearance2D;

namespace {

// Row-major index helper.
inline size_t idx(int x, int y, int w) { return static_cast<size_t>(y) * w + x; }

}  // namespace

// (1) Clearance is reported in METERS on a 0.1 m grid. An interior obstacle far
//     from the border makes its orthogonal neighbor read ~0.1 m and its
//     diagonal neighbor ~0.1414 m.
TEST(Clearance2DTest, ClearanceIsInMeters) {
  const int w = 11, h = 11;
  const double res = 0.1;
  std::vector<int8_t> map(static_cast<size_t>(w) * h, 0);  // all free
  // Single obstacle at the center (5,5): far from every border (5 cells away).
  map[idx(5, 5, w)] = 100;

  // Truncate generously so nothing near the center saturates.
  auto clr = computeClearanceField2D(w, h, res, map.data(), 1.0);
  ASSERT_EQ(clr.size(), static_cast<size_t>(w) * h);

  EXPECT_FLOAT_EQ(clr[idx(5, 5, w)], 0.0f);              // on the obstacle
  EXPECT_NEAR(clr[idx(4, 5, w)], 0.1f, 1e-4);            // one orthogonal cell
  EXPECT_NEAR(clr[idx(4, 4, w)], 0.1414f, 1e-3);         // one diagonal cell
}

// (2) The border ring is seeded to 0 (so OOB is treated as clearance 0). On a
//     free grid with no interior obstacle, edge cells read 0 and clearance
//     grows inward, capped by distance to the nearest border.
TEST(Clearance2DTest, BorderRingSeedsToZero) {
  const int w = 5, h = 5;
  const double res = 0.1;
  std::vector<int8_t> map(static_cast<size_t>(w) * h, 0);  // all free, no obstacle

  auto clr = computeClearanceField2D(w, h, res, map.data(), 1.0);

  // Every border cell is seeded at 0.
  for (int x = 0; x < w; ++x) {
    EXPECT_FLOAT_EQ(clr[idx(x, 0, w)], 0.0f);
    EXPECT_FLOAT_EQ(clr[idx(x, h - 1, w)], 0.0f);
  }
  for (int y = 0; y < h; ++y) {
    EXPECT_FLOAT_EQ(clr[idx(0, y, w)], 0.0f);
    EXPECT_FLOAT_EQ(clr[idx(w - 1, y, w)], 0.0f);
  }
  // Clearance grows inward from the seeded border: a cell one cell in from the
  // edge is ~0.1 m, and the 5x5 center (two cells in) is ~0.2 m.
  EXPECT_NEAR(clr[idx(1, 2, w)], 0.1f, 1e-4);
  EXPECT_NEAR(clr[idx(2, 2, w)], 0.2f, 1e-4);
}

// (3a) Projection moves a wall-adjacent goal outward to a cell with >= buffer
//      clearance.
TEST(Clearance2DTest, ProjectionMovesWallAdjacentGoal) {
  // A wide free field with a wall column at x=10 (interior). Grid is wide enough
  // that the interior is far from the map border.
  const int w = 41, h = 41;
  const double res = 0.1;
  std::vector<int8_t> map(static_cast<size_t>(w) * h, 0);
  for (int y = 0; y < h; ++y) map[idx(10, y, w)] = 100;  // vertical wall at x=10

  auto clr = computeClearanceField2D(w, h, res, map.data(), 2.0);

  // Goal sits right next to the wall (x=11), which has clearance ~0.1 m.
  const int gx = 11, gy = 20;
  ASSERT_LT(clr[idx(gx, gy, w)], 0.35f);

  const double buffer_m = 0.35;      // requires ~4 cells away from the wall
  const double search_radius_m = 3.0;
  int ox = -1, oy = -1;
  ASSERT_TRUE(projectToClearance2D(w, h, res, clr.data(), gx, gy, buffer_m, search_radius_m, ox,
                                   oy));
  // Projected cell must actually satisfy the buffer.
  EXPECT_GE(clr[idx(ox, oy, w)], static_cast<float>(buffer_m));
}

// (3b) Projection returns the input unchanged when the goal is already clear.
TEST(Clearance2DTest, ProjectionUnchangedWhenAlreadyClear) {
  const int w = 41, h = 41;
  const double res = 0.1;
  std::vector<int8_t> map(static_cast<size_t>(w) * h, 0);
  for (int y = 0; y < h; ++y) map[idx(10, y, w)] = 100;

  auto clr = computeClearanceField2D(w, h, res, map.data(), 2.0);

  // A goal well away from the wall and from all borders already clears a small
  // buffer.
  const int gx = 20, gy = 20;
  const double buffer_m = 0.35;
  ASSERT_GE(clr[idx(gx, gy, w)], static_cast<float>(buffer_m));

  int ox = -1, oy = -1;
  ASSERT_TRUE(projectToClearance2D(w, h, res, clr.data(), gx, gy, buffer_m, 3.0, ox, oy));
  EXPECT_EQ(ox, gx);
  EXPECT_EQ(oy, gy);
}

// (3c) Projection fails (returns false, no clear cell) when the goal is boxed in
//      and no cell within the search cap satisfies the buffer.
TEST(Clearance2DTest, ProjectionFailsWhenBoxedIn) {
  // Small fully-free grid: the border ring seeding means the largest achievable
  // clearance is bounded by distance to the edge. Request an impossible buffer.
  const int w = 7, h = 7;
  const double res = 0.1;
  std::vector<int8_t> map(static_cast<size_t>(w) * h, 0);

  auto clr = computeClearanceField2D(w, h, res, map.data(), 2.0);

  // Max clearance is at the center (3,3): 3 cells from the border = ~0.3 m.
  // Ask for 1.0 m, which no cell can satisfy -> projection must fail.
  int ox = 123, oy = 456;  // sentinels that must stay untouched on failure
  EXPECT_FALSE(projectToClearance2D(w, h, res, clr.data(), 3, 3, 1.0, 3.0, ox, oy));
  EXPECT_EQ(ox, 123);
  EXPECT_EQ(oy, 456);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
