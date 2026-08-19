/* ----------------------------------------------------------------------------
 * Copyright 2025, Kota Kondo, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Kota Kondo, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file clearance_2d.hpp
 * @brief Free-function 2D clearance field + goal projection for the ground-robot
 *        endpoint clearance buffer.
 *
 *  These are intentionally free functions operating on raw (width, height, res,
 *  const int8_t* map / const float* clearance) VIEWS -- no MapUtil (or PCL /
 *  Eigen / decomp) dependency -- so they are unit-testable in isolation. MapUtil
 *  wraps them (see buildClearance2D / getClearance2D / projectGoalToClearance2D).
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <vector>

namespace mighty {

/** @brief 8-connected truncated wavefront producing a clearance field in METERS.
 *
 *  Distance from each cell to the nearest NON-FREE cell, truncated at max_dist_m.
 *  Modeled on OccGrid2D::computeDistanceField, but with two seeding differences:
 *    1. Seed every cell where map2d[i] != free_val (this picks up occupied AND
 *       large-unknown-classified-as-occupied cells; the 2D map is binary).
 *    2. ALSO seed the entire border ring, so cells near the map edge read a
 *       small clearance and out-of-bounds is treated as clearance 0 (matching
 *       MapUtil::get2DOccupancy's conservative OOB == occupied). This is why
 *       computeDistanceField (which does NOT seed the border) cannot be reused.
 *
 *  @param width   Grid width in cells.
 *  @param height  Grid height in cells.
 *  @param res     Cell resolution [m].
 *  @param map2d   Row-major (x + width*y) binary occupancy view.
 *  @param max_dist_m Truncation distance [m]; untouched cells saturate here.
 *  @param free_val Sentinel value for a free cell (default 0).
 *  @return width*height clearance field in meters.
 */
inline std::vector<float> computeClearanceField2D(int width, int height, double res,
                                                  const int8_t* map2d, double max_dist_m,
                                                  int8_t free_val = 0) {
  const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
  const float INF = static_cast<float>(max_dist_m);
  std::vector<float> clearance(n, INF);
  if (width <= 0 || height <= 0 || map2d == nullptr) return clearance;

  // Seed all non-free cells AND the border ring at clearance 0.
  struct Cell {
    int x, y;
  };
  std::queue<Cell> q;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t i = static_cast<size_t>(y) * static_cast<size_t>(width) + x;
      const bool is_border = (x == 0 || x == width - 1 || y == 0 || y == height - 1);
      if (map2d[i] != free_val || is_border) {
        clearance[i] = 0.0f;
        q.push({x, y});
      }
    }
  }

  // 8-connected BFS with Euclidean step lengths (in meters).
  const int dx8[] = {-1, 0, 1, -1, 1, -1, 0, 1};
  const int dy8[] = {-1, -1, -1, 0, 0, 1, 1, 1};
  const float dd8[] = {1.414f, 1.0f, 1.414f, 1.0f, 1.0f, 1.414f, 1.0f, 1.414f};

  while (!q.empty()) {
    const Cell c = q.front();
    q.pop();
    const float cd = clearance[static_cast<size_t>(c.y) * static_cast<size_t>(width) + c.x];
    for (int i = 0; i < 8; ++i) {
      const int nx = c.x + dx8[i], ny = c.y + dy8[i];
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const float nd = cd + dd8[i] * static_cast<float>(res);
      if (nd >= max_dist_m) continue;
      const size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(width) + nx;
      if (nd < clearance[nidx]) {
        clearance[nidx] = nd;
        q.push({nx, ny});
      }
    }
  }
  return clearance;
}

/** @brief Outward (4-connected) BFS from a goal cell to the nearest cell whose
 *  clearance is >= buffer_m, capped at a search radius.
 *
 *  If the start cell already clears the buffer, returns it unchanged. If no
 *  qualifying cell is found within the cap (or on bad input), returns false and
 *  leaves the outputs untouched -- callers keep the raw goal (never drop it).
 *
 *  @param width          Grid width in cells.
 *  @param height         Grid height in cells.
 *  @param res            Cell resolution [m].
 *  @param clearance      Row-major clearance field [m] (from computeClearanceField2D).
 *  @param gx,gy          Goal cell indices.
 *  @param buffer_m       Required clearance [m].
 *  @param search_radius_m Maximum outward search radius [m].
 *  @param out_x,out_y    Output projected cell (written only on success).
 *  @return true if a qualifying cell was found (or the start already cleared).
 */
inline bool projectToClearance2D(int width, int height, double res, const float* clearance, int gx,
                                 int gy, double buffer_m, double search_radius_m, int& out_x,
                                 int& out_y) {
  if (width <= 0 || height <= 0 || clearance == nullptr) return false;

  const float buf = static_cast<float>(buffer_m);
  auto clear_at = [&](int x, int y) -> float {
    // Out-of-bounds reads clearance 0 (conservative, matches get2DOccupancy OOB).
    if (x < 0 || x >= width || y < 0 || y >= height) return 0.0f;
    return clearance[static_cast<size_t>(y) * static_cast<size_t>(width) + x];
  };

  // Already clear (or goal cell OOB but somehow clear): keep the goal cell.
  if (clear_at(gx, gy) >= buf) {
    out_x = gx;
    out_y = gy;
    return true;
  }

  const int max_r = (res > 0.0) ? static_cast<int>(std::ceil(search_radius_m / res)) : 0;

  std::vector<char> visited(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
  struct Cell {
    int x, y;
  };
  std::queue<Cell> q;
  if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
    visited[static_cast<size_t>(gy) * static_cast<size_t>(width) + gx] = 1;
    q.push({gx, gy});
  }

  const int dx4[] = {1, -1, 0, 0};
  const int dy4[] = {0, 0, 1, -1};
  while (!q.empty()) {
    const Cell c = q.front();
    q.pop();
    for (int i = 0; i < 4; ++i) {
      const int nx = c.x + dx4[i], ny = c.y + dy4[i];
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      // Cap the outward search radius (cell distance from the goal cell).
      if (std::abs(nx - gx) > max_r || std::abs(ny - gy) > max_r) continue;
      const size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(width) + nx;
      if (visited[nidx]) continue;
      visited[nidx] = 1;
      if (clear_at(nx, ny) >= buf) {
        out_x = nx;
        out_y = ny;
        return true;
      }
      q.push({nx, ny});
    }
  }
  return false;
}

}  // namespace mighty
