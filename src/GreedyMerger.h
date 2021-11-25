#ifndef FSG_GREEDYMERGER_H_
#define FSG_GREEDYMERGER_H_

#include <memory>
#include <utility>
#include <vector>
#include "FastAcontrarioValidator.h"
#include "SegmentsGroup.h"

namespace upm {
/**
 * Implements the "Segments Groups Proposer" algorithm from:
 *   @cite Suárez, I., Muñoz, E., Buenaposada, J. M., & Baumela, L. (2018).
 *   FSG: A statistical approach to line detection via fast segments grouping. In Proc. IROS(pp. 97-102).
 *
 * Given a set of already detected segments in an image, the method find those that are well aligned.
 */
class GreedyMerger {
 public:
  explicit GreedyMerger(cv::Size imgSize);

  /**
   * @brief Merge segments that below to the same line
   * @param originalSegments The original segments to merge
   * @param mergedLines The output merged lines
   * @param assignations A vector where each element is a vector containing
   * the indices of the originalSegments segments associated to each element
   * of mergedLines. This vector have the same size of mergedLines.
   */
  void mergeSegments(const Segments &originalSegments, Segments &mergedLines, SegmentClusters &assignations);

  // Getters and setters
  void setImageSize(const cv::Size &size);
  void setNumOrientationBins(int numOrientationBins);
  void setNumLengthBins(int numLengthBins);
  void setEndPointError(float endpointError);
  void setMinSegSize(double minSegSize);
  void setEndpointsMaxDistance(double maxDistance);
  void setSegmentsValidator(std::shared_ptr<FastAcontrarioValidator> validator);

  /**
   * @brief Builds an histograms of segments by its angle. The histogram will be used to find merging candidates.
   * @param segments The input line segments
   * @param bins The number of bins in the histogram. A lot of bins will make the method faster but some
   * candidates can be missing. With few bins the method will consider more candidates and the computational
   * cost will be higher.
   * @return
   */
  static SegmentClusters getOrientationHistogram(const Segments &segments, int bins = 90);

  /**
   * @brief Partial sort the segments by length
   * @param segments The input line segments
   * @param bins The number of bins used in the internal histogram. The bigger the more precise the sorting.
   * @param imgSize The image size
   * @return List of indices that sort the list of segments from larger to smaller.
   */
  static std::vector<unsigned int> partialSortByLength(const Segments &segments, int bins, const cv::Size &imgSize);

  /**
   * Calculates a measure of the distance between the base segment and a new candidate segment.
   * @param base_seg The base segment
   * @param seg The candidate segment
   * @return The minimum distance between the endpoints of the base and
   * candidate segments / the mean of the candidate segment length and the
   * average length of the segments in the base segment.
   */
  static inline double nearestEndpointDist(SegmentsGroup &base_seg, const Segment &seg);  // NOLINT

  /**
   * Returns the line equations of the lines that define the conic where we
   * are going to look for the segments that are in the same line that seg.
   * @param seg The segment in format [x1, y1, x2, y2] where (x1, y1) and
   * (x2, y2) are the segment endpoints.
   * @param radius The radius of error that a segment endpoint can have.
   * We use it to calculate the width of the conic.
   * @return A pair {l1, l2} where l1 and l2 are the general line equations
   * in format ax + by = c of the lines that define a conic around the
   * segment seg given a circle of radius radius
   */
  static std::pair<cv::Vec3f, cv::Vec3f> getTangentLineEqs(Segment seg, float radius);

 private:
  // Number of orientations bins used to sort the segments. The segments with
  // an angular difference greater than 180 / N_ORIENTATION_BINS degrees
  // wouldn't be merged in the same line
  int N_ORIENTATION_BINS = 50;
  // Number of length bins used to sort the segments by its length
  int N_LENGTH_BINS = 1000;
  float ENDPOINT_ERROR = 6;  // In pixels
  bool DO_NOT_MERGE_OVERLAPPING_SEGS = true;
  double MIN_SEGMENT_SIZE = 0.01;  // In percentage of the image
  double ENDPOINTS_MAX_DISTANCE = 4;
  cv::Size mImgSize = cv::Size(800, 480);
  std::shared_ptr<FastAcontrarioValidator> mValidator;
};

}  // namespace upm

#endif  // FSG_GREEDYMERGER_H_
