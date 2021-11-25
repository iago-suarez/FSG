/**
 * @copyright 2018 Xoan Iago Suarez Canosa. All rights reserved.
 * Constact: iago.suarez.canosa@alumnos.upm.es
 * Software developed in the PhD: Augmented Reality for Urban Environments
 */
#include <gtest/gtest.h>

int main(int argc, char **argv) {

  // Init an run the regression tests
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}