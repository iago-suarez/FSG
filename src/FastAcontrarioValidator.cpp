#include "FastAcontrarioValidator.h"
#include "Utils.h"

#define UPM_DEGREES_TO_RADS 0.01745329252  // (M_PI / 180.0)

namespace upm {

/**
 * @brief calculates the integral of the function:
 * \int asin\left(\frac{\rho + x}{r}\right) + asin\left(\frac{\rho - x}{r}\right) dx
 * @param x The variable
 * @param p rho value
 * @param r radio value
 * @return
 */
static inline double int_step_1(double x, double p, double r) {
  double val1 = +(x + p) * std::asin((x + p) / r);
  double val2 = -(x - p) * std::asin((x - p) / r);
  double val3 = +r * std::sqrt(1. - ((x + p) * (x + p)) / (r * r));
  double val4 = -r * std::sqrt(1. - ((x - p) * (x - p)) / (r * r));
  return val1 + val2 + val3 + val4;
}

/**
 * @brief calculates the integral of the function:
 * \int \frac{\pi}{2} + asin\left(\frac{\rho - x}{r}\right) dx
 * @param x The variable
 * @param p rho value
 * @param r radio value
 * @return
 */
static inline double int_step_2(double x, double p, double r) {
  return -(x - p) * std::asin((x - p) / r)
      - r * std::sqrt(1 - ((x - p) * (x - p)) / (r * r))
      + M_PI * x / 2.0;
}

/**
 * @brief calculates the integral of the function:
 * \int \frac{\pi}{2} + \frac{\pi}{2} dx
 * @param x The variable
 * @param p rho value
 * @param r radio value
 * @return
 */
static inline double int_step_3(double x, double p, double r) {
  return M_PI * x;
}

/**
 * Computes the logarithm of the combinations of N elements taken k by k.
 * @param N elements
 * @param k taken k by k
 * @return
 */
static inline double log_combinations(uint64 N, uint64 k) {
  return lgamma(N + 1) - (lgamma(k + 1) + lgamma(N - k + 1));
}

/**
 * @brief calculates the integral by parts of the function ( Eq. 3 of the paper):
 * \int_0^{\rho} asin \left( min \left(1,  \frac{\rho + x}{r}\right) \right)\pi + asin\left( min \left(1, \frac{\rho - x}{r}\right)\right) dx
 * Between 0 and rho.
 * @param p The rho value, i.e. half of the width of the rectangle
 * @param r The radio value, i.e. The length of the segment we want to calculate the probability.
 * @return
 */
static double integral_of_x_p_r(double p, double r) {
  if (r > 2 * p) {
    return (int_step_1(p, p, r) - int_step_1(0, p, r)) * M_1_PI;
  } else if (r >= p) {
    double area_1 = int_step_1(r - p, p, r) - int_step_1(0, p, r);
    double area_2 = int_step_2(p, p, r) - int_step_2(r - p, p, r);
    return (area_1 + area_2) / M_PI;
  } else {
    // ( r < p)
    double area_3 = int_step_3(p - r, p, r) - int_step_3(0, p, r);
    double area_2 = int_step_2(p, p, r) - int_step_2(p - r, p, r);
    return (area_3 + area_2) / M_PI;
  }
}

double FastAcontrarioValidator::fastLogCombinations(int64_t N, int64_t k) const {
  assert((N > 0) & (k > 0));
  if (k <= CACHED_LOG_COMB_SIZE_K) {
    auto it = mCachedLogComb.find(N);
    if (it != mCachedLogComb.end())
      return it->second[k];
  }

  return log_combinations(N, k);
}

void FastAcontrarioValidator::cachLogCombinations(int N) {
  // If the entry already exists do nothing
  auto it = mCachedLogComb.find(N);
  if (it != mCachedLogComb.end()) return;

  std::vector<double> entry(CACHED_LOG_COMB_SIZE_K + 1);
  for (int i = 1; i <= CACHED_LOG_COMB_SIZE_K; i++) {
    entry[i] = log_combinations(N, i);
  }
  mCachedLogComb.insert({N, entry});
}

void FastAcontrarioValidator::cachLogCombinationsForImage(const cv::Size &imgSize) {
  mDomainSize = imgSize;
  int N = MIN_SEG_DENSITY * imgSize.area();
  cachLogCombinations(N);
}

double FastAcontrarioValidator::candidateValidity(const SegmentsGroupPtr& group,
                                                  const Segments &segments,
                                                  unsigned int candidate) {
  assert(mDomainSize.area() > 0);

  // Impose a minimum in the number of segments
  const double N = MIN_SEG_DENSITY * mDomainSize.area();
  const double k = group->getSegmentIds().size() + 1;
  // Use a rectangle to calculate the line to with we want to measure the distance

  cv::RotatedRect R = group->fitRectangleToCandidate(candidate);

  // If the rectangle has size 0 its probability is inf
  if (R.size.area() == 0) return -DBL_MAX;
  double len = R.size.height;
  double rho = R.size.width;
  double prod_prob_2nd = 1;

  for (const unsigned int i: group->getSegmentIds()) {
    double radio = segLength(segments[i]);
    double prob_seg = integral_of_x_p_r(rho, radio);
    prod_prob_2nd *= prob_seg;
  }
  // Calculate the values for the candidate
  double radio = segLength(segments[candidate]);
  double prob_seg = integral_of_x_p_r(rho, radio);
  prod_prob_2nd *= prob_seg;

  double prod_prob_1st = std::pow((rho * len) / mDomainSize.area(), k);
  double prob_h0 = prod_prob_1st * prod_prob_2nd;
  // As to calculate the number of combinations we need to calculate the
  // factorial and here the factorial is of quantities so large that we can
  // not store in a double variable, we are going to calculate the neperian
  // logarithm of the NFA instead of the NFA.

  double logNtests = fastLogCombinations(N, k);
  double logNFA = logNtests + std::log(prob_h0);
  return logNFA;
}

void FastAcontrarioValidator::setDomainSize(const cv::Size &domainSize) {
  mDomainSize = domainSize;
}

}
