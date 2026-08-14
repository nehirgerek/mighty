/* ----------------------------------------------------------------------------
 * Diagnostic terrain height-gap geometry (ROS-free). See terrain_gap.hpp.
 * -------------------------------------------------------------------------- */
#include "mighty/terrain_gap.hpp"

#include <algorithm>
#include <cmath>

namespace mighty {

const char* terrainGapClassStr(TerrainGapClass c) {
  switch (c) {
    case TerrainGapClass::ONE_SIDED: return "ONE_SIDED";
    case TerrainGapClass::SAME_LEVEL: return "SAME_LEVEL";
    case TerrainGapClass::HEIGHT_DIFFERENCE: return "HEIGHT_DIFFERENCE";
    case TerrainGapClass::INSUFFICIENT_EVIDENCE: return "INSUFFICIENT_EVIDENCE";
  }
  return "?";
}

namespace {
enum class Cell { FREE, UNKNOWN, OCCUPIED, OOB };
Cell cellAt(const OccGrid2D& occ, int ix, int iy) {
  if (!occ.inBounds(ix, iy)) return Cell::OOB;
  if (occ.isOccupied(ix, iy)) return Cell::OCCUPIED;
  if (occ.isUnknown(ix, iy)) return Cell::UNKNOWN;
  return Cell::FREE;
}
double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
}  // namespace

std::vector<TerrainGapPair> findGapPairs(const OccGrid2D& occ,
                                         const std::vector<Eigen::Vector2d>& frontier_cells,
                                         const TerrainGapParams& P, bool* had_unknown_neighbour) {
  std::vector<TerrainGapPair> pairs;
  if (had_unknown_neighbour) *had_unknown_neighbour = false;
  const double res = occ.resolution();
  if (res <= 1e-6) return pairs;
  const double step = 0.5 * res;                       // <= half a cell (don't skip a thin cell)
  const double max_w = P.max_gap_width_m;
  const int dx8[] = {1, 1, 0, -1, -1, -1, 0, 1};
  const int dy8[] = {0, 1, 1, 1, 0, -1, -1, -1};

  for (const auto& A : frontier_cells) {
    int ax, ay;
    occ.worldToGrid(A.x(), A.y(), ax, ay);
    if (!occ.inBounds(ax, ay)) continue;

    // Mean direction into UNKNOWN from the 8-neighbourhood.
    Eigen::Vector2d d = Eigen::Vector2d::Zero();
    int n_unk = 0;
    for (int k = 0; k < 8; ++k) {
      if (cellAt(occ, ax + dx8[k], ay + dy8[k]) == Cell::UNKNOWN) {
        double wx, wy;
        occ.gridToWorld(ax + dx8[k], ay + dy8[k], wx, wy);
        d += Eigen::Vector2d(wx - A.x(), wy - A.y());
        ++n_unk;
      }
    }
    if (n_unk == 0) continue;  // not adjacent to UNKNOWN -> skip this cell
    if (had_unknown_neighbour) *had_unknown_neighbour = true;
    if (d.norm() < 1e-9) continue;
    d.normalize();

    // March along d: require entering a contiguous UNKNOWN run, accept B at the first
    // FREE after it; reject on OCCUPIED / OOB / run longer than max_gap_width.
    bool entered_unknown = false;
    int last_ix = ax, last_iy = ay;
    for (double dist = step; dist <= max_w + 1e-9; dist += step) {
      const Eigen::Vector2d p = A + dist * d;
      int ix, iy;
      occ.worldToGrid(p.x(), p.y(), ix, iy);
      if (ix == last_ix && iy == last_iy) continue;  // same cell, keep marching
      last_ix = ix;
      last_iy = iy;
      const Cell c = cellAt(occ, ix, iy);
      if (c == Cell::OCCUPIED || c == Cell::OOB) break;  // reject pair
      if (c == Cell::UNKNOWN) {
        entered_unknown = true;
        continue;
      }
      // c == FREE
      if (!entered_unknown) continue;  // still on side A's free run
      double bx, by;
      occ.gridToWorld(ix, iy, bx, by);
      TerrainGapPair pr;
      pr.a_world = A;
      pr.b_world = Eigen::Vector2d(bx, by);
      pr.gap_width_m = (pr.b_world - A).norm();
      pairs.push_back(pr);
      break;
    }
  }
  return pairs;
}

TerrainGapResult classifyTerrainGap(const std::vector<TerrainGapPair>& valid_pairs,
                                    const TerrainGapParams& P) {
  TerrainGapResult r;
  r.pairs = valid_pairs;
  if (static_cast<int>(valid_pairs.size()) < P.min_pairs) {
    r.classification = TerrainGapClass::INSUFFICIENT_EVIDENCE;
    return r;
  }
  std::vector<double> dz, absdz, width;
  dz.reserve(valid_pairs.size());
  for (const auto& p : valid_pairs) {
    dz.push_back(p.dz);
    absdz.push_back(std::abs(p.dz));
    width.push_back(p.gap_width_m);
  }
  r.median_dz = median(dz);
  r.median_abs_dz = median(absdz);
  r.median_gap_width_m = median(width);
  std::vector<double> dev;
  dev.reserve(dz.size());
  for (double v : dz) dev.push_back(std::abs(v - r.median_dz));
  r.mad_dz = median(dev);
  r.classification = (std::abs(r.median_dz) >= P.height_threshold_m)
                         ? TerrainGapClass::HEIGHT_DIFFERENCE
                         : TerrainGapClass::SAME_LEVEL;
  return r;
}

}  // namespace mighty
