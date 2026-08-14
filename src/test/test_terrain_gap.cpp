/* ----------------------------------------------------------------------------
 * ROS-free tests for the diagnostic terrain height-gap geometry (terrain_gap.*).
 * Uses OccGrid2D::fromTristate (nav_msgs only) -- no full ROS graph.
 * -------------------------------------------------------------------------- */
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "mighty/occ_grid_2d.hpp"
#include "mighty/terrain_gap.hpp"

using namespace mighty;

namespace {
constexpr int8_t FREE = 0, OCC = 100, UNK = -1;

// Build a 20x5 grid (res 0.15, origin 0,0); column state from a lambda of ix.
std::shared_ptr<const OccGrid2D> makeGrid(std::function<int8_t(int)> colState) {
  const int nx = 20, ny = 5;
  std::vector<int8_t> data(nx * ny, FREE);
  for (int iy = 0; iy < ny; ++iy)
    for (int ix = 0; ix < nx; ++ix) data[iy * nx + ix] = colState(ix);
  return OccGrid2D::fromTristate(nx, ny, 0.15, 0.0, 0.0, data);
}
// Frontier A cells at the last FREE column before the UNKNOWN run (ix=4), a few rows.
std::vector<Eigen::Vector2d> aCells() {
  std::vector<Eigen::Vector2d> c;
  for (int iy = 1; iy <= 3; ++iy) c.emplace_back((4 + 0.5) * 0.15, (iy + 0.5) * 0.15);
  return c;
}
}  // namespace

// FREE | UNKNOWN run | FREE(B) -> a valid A/B pair with gap width ~0.75 m.
TEST(TerrainGap, FindsFarFreeBoundary) {
  auto g = makeGrid([](int ix) { return (ix >= 5 && ix <= 8) ? UNK : FREE; });
  TerrainGapParams P;
  bool had_unk = false;
  auto pairs = findGapPairs(*g, aCells(), P, &had_unk);
  EXPECT_TRUE(had_unk);
  ASSERT_GT(pairs.size(), 0u);
  for (const auto& pr : pairs) {
    EXPECT_GT(pr.b_world.x(), pr.a_world.x());   // B is past the UNKNOWN run (+x)
    EXPECT_NEAR(pr.gap_width_m, 0.75, 0.16);     // ~ (9-4) cells * 0.15
  }
}

// A -> UNKNOWN -> OCCUPIED : rejected (no far FREE boundary).
TEST(TerrainGap, OccupiedBeyondUnknownRejected) {
  auto g = makeGrid([](int ix) {
    if (ix >= 5 && ix <= 8) return UNK;
    if (ix == 9) return OCC;
    return FREE;
  });
  TerrainGapParams P;
  auto pairs = findGapPairs(*g, aCells(), P, nullptr);
  EXPECT_EQ(pairs.size(), 0u);
}

// A -> UNKNOWN -> out of bounds (UNKNOWN to the edge): rejected.
TEST(TerrainGap, UnknownToBoundaryRejected) {
  auto g = makeGrid([](int ix) { return (ix >= 5) ? UNK : FREE; });  // unknown to +x edge
  TerrainGapParams P;
  auto pairs = findGapPairs(*g, aCells(), P, nullptr);
  EXPECT_EQ(pairs.size(), 0u);
}

// Gap wider than max_gap_width_m: rejected.
TEST(TerrainGap, GapTooWideRejected) {
  auto g = makeGrid([](int ix) { return (ix >= 5 && ix <= 8) ? UNK : FREE; });
  TerrainGapParams P;
  P.max_gap_width_m = 0.3;  // 0.75 m gap exceeds this
  auto pairs = findGapPairs(*g, aCells(), P, nullptr);
  EXPECT_EQ(pairs.size(), 0u);
}

// No UNKNOWN neighbour -> nothing to pair, had_unknown_neighbour false.
TEST(TerrainGap, NoUnknownNeighbour) {
  auto g = makeGrid([](int) { return FREE; });  // all free
  TerrainGapParams P;
  bool had_unk = true;
  auto pairs = findGapPairs(*g, aCells(), P, &had_unk);
  EXPECT_FALSE(had_unk);
  EXPECT_EQ(pairs.size(), 0u);
}

// Classification thresholds (diagnostic only).
TEST(TerrainGap, Classification) {
  TerrainGapParams P;  // min_pairs=5, height_threshold=0.05
  auto mk = [](double dz) { TerrainGapPair p; p.dz = dz; p.gap_width_m = 0.3; return p; };

  std::vector<TerrainGapPair> few = {mk(-0.2), mk(-0.2), mk(-0.2), mk(-0.2)};  // 4 < 5
  EXPECT_EQ(classifyTerrainGap(few, P).classification, TerrainGapClass::INSUFFICIENT_EVIDENCE);

  std::vector<TerrainGapPair> big;
  for (int i = 0; i < 6; ++i) big.push_back(mk(-0.168));
  auto rb = classifyTerrainGap(big, P);
  EXPECT_EQ(rb.classification, TerrainGapClass::HEIGHT_DIFFERENCE);
  EXPECT_NEAR(rb.median_dz, -0.168, 1e-9);
  EXPECT_NEAR(rb.median_abs_dz, 0.168, 1e-9);

  std::vector<TerrainGapPair> flat;
  for (int i = 0; i < 6; ++i) flat.push_back(mk(0.01));  // below 0.05
  EXPECT_EQ(classifyTerrainGap(flat, P).classification, TerrainGapClass::SAME_LEVEL);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
