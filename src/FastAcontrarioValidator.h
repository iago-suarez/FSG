#ifndef FSG_FASTACONTRARIOVALIDATOR_H_
#define FSG_FASTACONTRARIOVALIDATOR_H_

#include <opencv2/opencv.hpp>

#include <map>
#include <algorithm>
#include <vector>

#include "SegmentsGroup.h"
#include "Utils.h"

namespace upm {

/**
 * Implements the "Probabilistic Segments Group Validation" from:
 *   @cite Suárez, I., Muñoz, E., Buenaposada, J. M., & Baumela, L. (2018).
 *   FSG: A statistical approach to line detection via fast segments grouping. In Proc. IROS(pp. 97-102).
 * The class takes a group of segments and returns its validity based on the NFA (Number of False Alarms).
 * Note that we use log(NFA) instead of NFA because the huge number of combinations in Eq. (1) and the
 * tiny probability make the method numerically unstable.
 */
class FastAcontrarioValidator {
 public:
  /**
   * @brief Computes the candidate logarithm of the NFA (Number of False Alarms)
   * @param group The base segment to which we want to add the candidate.
   * @param segments The full list of segments detected in the image
   * @param candidate The index in the vector of segments of the candidate
   * @return The logarithm of the number of false alarms.
   */
  double candidateValidity(const SegmentsGroupPtr &group, const Segments &segments, unsigned int candidate);

  /**
   * Sets the image size, used to compute internal probabilities and creates a cache
   * with the logarithm of the combinations (s c) fro equation (1) of the paper.
   * @param imgSize The image size
   */
  void cachLogCombinationsForImage(const cv::Size &imgSize);

  // Getters and setters
  inline double getEpsilon() const { return EPSILON; }

 private:
  /**
   * @brief Calculate the logarithm of the number of combinations of N elements taken k by k.
   * @param N Combinations of N elements
   * @param k Taken k by k
   * @return
   */
  double fastLogCombinations(int64_t N, int64_t k) const;

  /**
   * Pre-computes a cache with the log of the number of combinations.
   * @param N Number of segments detected in the image, or an upper bound of it.
   */
  void cachLogCombinations(int N);

  double EPSILON = 0;
  // Suppose a minimum of one segment by each 1000px^2
  double MIN_SEG_DENSITY = 1 / 150.0;
  int CACHED_LOG_COMB_SIZE_K = 20;
  std::map<int, std::vector<double> > mCachedLogComb;
  cv::Size mDomainSize;
};
}  // namespace upm

#endif  // FSG_FASTACONTRARIOVALIDATOR_H_
