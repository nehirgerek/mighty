/* ----------------------------------------------------------------------------
 * Scoped, per-goal terminal-radius override for MIGHTY.
 *
 * The frontier-viewpoint state machine needs core MIGHTY to keep planning/controlling
 * until the active viewpoint terminal point (q_pre or q*) is within a tight tolerance
 * (arrival_tol_m, e.g. 0.25 m), rather than stopping at the generic hardware
 * goal_radius (e.g. 1.0 m). This holds an OPTIONAL override that, when active, replaces
 * par_.goal_radius ONLY in the terminal-goal completion checks inside
 * MIGHTY::needReplan(). When inactive, the generic radius is used unchanged.
 *
 * Thread-safe (lock-free): written from the node callback group, read from the planning
 * loop. A NaN sentinel means "no override" so the whole state fits one atomic<double>.
 * -------------------------------------------------------------------------- */
#pragma once

#include <atomic>
#include <cmath>
#include <limits>

namespace mighty {

class GoalRadiusOverride {
 public:
  /** @brief Activate the override with the given terminal radius [m]. */
  void set(double radius) { value_.store(radius, std::memory_order_relaxed); }

  /** @brief Deactivate the override; the generic radius is used again. */
  void clear() { value_.store(std::numeric_limits<double>::quiet_NaN(), std::memory_order_relaxed); }

  /** @brief True while an override radius is active. */
  bool active() const { return !std::isnan(value_.load(std::memory_order_relaxed)); }

  /** @brief The active terminal radius: the override if set, else @p generic. */
  double radius(double generic) const {
    const double v = value_.load(std::memory_order_relaxed);
    return std::isnan(v) ? generic : v;
  }

 private:
  std::atomic<double> value_{std::numeric_limits<double>::quiet_NaN()};
};

}  // namespace mighty
