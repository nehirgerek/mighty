/* ----------------------------------------------------------------------------
 * Tests for the scoped terminal-goal radius override used by frontier-viewpoint
 * navigation. ROS-free: exercises mighty::GoalRadiusOverride and the two predicates
 * it feeds (MIGHTY::needReplan terminal completion, node arrival transition).
 * -------------------------------------------------------------------------- */
#include <gtest/gtest.h>

#include "mighty/goal_radius_override.hpp"

using mighty::GoalRadiusOverride;

namespace {
// Mirrors MIGHTY::needReplan terminal completion: reached iff dist < active radius.
bool terminalReached(const GoalRadiusOverride& o, double generic, double dist) {
  return dist < o.radius(generic);
}
// Mirrors the node arrival transition: within arrival tolerance (inclusive).
bool arrived(double dist, double tol) { return dist <= tol; }
}  // namespace

// A normal goal (no override) completes at the generic radius.
TEST(GoalRadiusOverride, NormalGoalUsesGenericRadius) {
  GoalRadiusOverride o;
  EXPECT_FALSE(o.active());
  EXPECT_DOUBLE_EQ(o.radius(1.0), 1.0);
  EXPECT_TRUE(terminalReached(o, 1.0, 0.90));
  EXPECT_FALSE(terminalReached(o, 1.0, 1.10));
}

// Override = 0.25 must NOT report GOAL_REACHED anywhere in 0.30..0.99 m.
TEST(GoalRadiusOverride, OverrideNotReachedBetween030And099) {
  GoalRadiusOverride o;
  o.set(0.25);
  EXPECT_TRUE(o.active());
  EXPECT_DOUBLE_EQ(o.radius(1.0), 0.25);
  for (double d : {0.30, 0.50, 0.75, 0.99}) {
    EXPECT_FALSE(terminalReached(o, 1.0, d)) << "d=" << d;
  }
}

// It DOES report reached below 0.25 m.
TEST(GoalRadiusOverride, OverrideReachedBelow025) {
  GoalRadiusOverride o;
  o.set(0.25);
  EXPECT_TRUE(terminalReached(o, 1.0, 0.24));
  EXPECT_TRUE(terminalReached(o, 1.0, 0.10));
  EXPECT_FALSE(terminalReached(o, 1.0, 0.25));  // strict '<', matches needReplan
}

// Clearing the override restores the generic radius.
TEST(GoalRadiusOverride, ClearRestoresGeneric) {
  GoalRadiusOverride o;
  o.set(0.25);
  o.clear();
  EXPECT_FALSE(o.active());
  EXPECT_DOUBLE_EQ(o.radius(1.0), 1.0);
  EXPECT_TRUE(terminalReached(o, 1.0, 0.90));  // generic behavior again
}

// A finished/invalidated viewpoint must not leave the override active for the next goal.
TEST(GoalRadiusOverride, NoLeakIntoNextGoal) {
  GoalRadiusOverride o;
  o.set(0.25);   // viewpoint navigation active
  o.clear();     // endViewpointObservation()
  EXPECT_FALSE(o.active());
  EXPECT_DOUBLE_EQ(o.radius(1.0), 1.0);
  EXPECT_TRUE(terminalReached(o, 1.0, 0.90));  // next ordinary goal sees generic radius
}

// APPROACH_PRE / APPROACH_Q transition only within the arrival tolerance.
TEST(GoalRadiusOverride, ApproachTransitionsOnlyWithinTol) {
  const double tol = 0.25;
  EXPECT_FALSE(arrived(0.30, tol));  // does not transition at 0.30
  EXPECT_FALSE(arrived(0.26, tol));
  EXPECT_TRUE(arrived(0.25, tol));   // inclusive boundary
  EXPECT_TRUE(arrived(0.10, tol));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
