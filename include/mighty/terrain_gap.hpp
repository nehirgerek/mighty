/* ----------------------------------------------------------------------------
 * Diagnostic 2.5-D height-difference check for the selected exploration frontier.
 *
 * READ-ONLY / OBSERVATION ONLY. This never changes the selected frontier, the goal,
 * occupancy, ESDF, visitation, or any planner/controller behavior. It pairs the
 * selected frontier's known-FREE edge (side A) with a known-FREE cell on the far side
 * of the adjacent UNKNOWN run (side B), purely as geometric/elevation evidence, and
 * reports whether the two observed surfaces differ in elevation.
 *
 * The boundary-pairing geometry here is ROS-free (operates on OccGrid2D + STL). The
 * elevation sampling / TF / markers live in mighty_node (they need grid_map + tf2).
 * -------------------------------------------------------------------------- */
#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include "mighty/occ_grid_2d.hpp"

namespace mighty {

struct TerrainGapParams {
  double max_gap_width_m   = 1.0;   // reject a ray whose UNKNOWN run exceeds this
  double height_threshold_m = 0.05;  // |median_dz| >= this -> HEIGHT_DIFFERENCE (diagnostic)
  int    min_pairs         = 5;      // minimum valid A/B elevation pairs to classify
  double sample_radius_m   = 0.15;   // elevation patch radius (used node-side)
};

// One A->UNKNOWN...->B geometric pairing. z_*/variance_*/dz are filled node-side after
// elevation sampling (NaN until then).
struct TerrainGapPair {
  Eigen::Vector2d a_world = Eigen::Vector2d::Zero();
  Eigen::Vector2d b_world = Eigen::Vector2d::Zero();
  double gap_width_m = 0.0;
  double z_a = std::numeric_limits<double>::quiet_NaN();
  double z_b = std::numeric_limits<double>::quiet_NaN();
  double variance_a = std::numeric_limits<double>::quiet_NaN();
  double variance_b = std::numeric_limits<double>::quiet_NaN();
  double dz = std::numeric_limits<double>::quiet_NaN();  // z_b - z_a
};

enum class TerrainGapClass {
  ONE_SIDED,             // frontier had UNKNOWN neighbours but no far-side FREE boundary
  SAME_LEVEL,            // enough pairs, |median_dz| < threshold
  HEIGHT_DIFFERENCE,     // enough pairs, |median_dz| >= threshold
  INSUFFICIENT_EVIDENCE  // too few valid elevation pairs
};
const char* terrainGapClassStr(TerrainGapClass c);

struct TerrainGapResult {
  TerrainGapClass classification = TerrainGapClass::INSUFFICIENT_EVIDENCE;
  std::vector<TerrainGapPair> pairs;   // only the elevation-valid pairs
  double median_dz = 0.0;
  double median_abs_dz = 0.0;
  double median_gap_width_m = 0.0;
  double mad_dz = 0.0;                  // median(|dz - median_dz|)
};

/** @brief Geometric A->UNKNOWN-run->B pairing over the CURRENT occupancy grid for the
 *  selected frontier's cells. ROS-free. For each frontier cell with >=1 UNKNOWN
 *  8-neighbour, casts a ray along the mean direction into UNKNOWN (spacing <= 0.5*res),
 *  requires entering a contiguous UNKNOWN run, and accepts B at the first FREE cell after
 *  the run. Rejects on OCCUPIED, OOB, or run length > max_gap_width_m (one-sided). The
 *  returned pairs carry only geometry; elevations are filled by the caller.
 *  @param had_unknown_neighbour set true if any frontier cell had an UNKNOWN neighbour
 *         (lets the caller distinguish ONE_SIDED from "not a real frontier"). */
std::vector<TerrainGapPair> findGapPairs(const OccGrid2D& occ,
                                         const std::vector<Eigen::Vector2d>& frontier_cells,
                                         const TerrainGapParams& P, bool* had_unknown_neighbour);

/** @brief Robust classification from elevation-valid pairs (dz set). ROS-free.
 *  < min_pairs -> INSUFFICIENT_EVIDENCE; else HEIGHT_DIFFERENCE if
 *  |median_dz| >= height_threshold_m, else SAME_LEVEL. Fills medians + MAD. */
TerrainGapResult classifyTerrainGap(const std::vector<TerrainGapPair>& valid_pairs,
                                    const TerrainGapParams& P);

}  // namespace mighty
