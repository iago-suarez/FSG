#ifndef FSG_SEGMENTSGROUP_H_
#define FSG_SEGMENTSGROUP_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "Utils.h"

namespace upm {

/**
 * This class represent a group of aligned segments.
 * The fitting is done with a classic least square fitter that selects the best direction to do
 * the fit (vertical of horizontal).
 */
class SegmentsGroup {
 public:
  /**
   * Creates the group of line allImgSegs.
   * @param allImgSegs Pointer to all the allImgSegs detected in the image.
   * The ones belonging to the group should be later added with addNewSegment
   */
  explicit SegmentsGroup(const Segments *allImgSegs) : mAllImgSegments(allImgSegs) {}

  /**
   * Adds a new line segment to the group. The integer should index mAllImgSegments.
   * @param newSegmentId
   */
  void addNewSegment(unsigned int newSegmentId);

  /**
   * @brief Returns the line equation that best fits with the segments group.
   * @return the line equation as: ax + by + c = 0
   */
  cv::Vec3d getLineEquation() const;

  /**
   * @brief Return the endpoints of the base segment that represents all the contained segments.
   * @return The endpoints in format x0, y0, x1, y1
   */
  cv::Vec4d getBaseSeg() const;

  /**
   * @return Returns the list of segments in the group.
   */
  const std::vector<unsigned int> &getSegmentIds() const;

  /**
   * @brief Returns true if the segment with index seg_idx overlaps more that a 20%
   * @param seg_idx The index of the segment we want to check
   * @param max_overlap_allowed The maximum amount of overlap
   * @return true if the segment overlaps and false otherwise
   */
  bool overlaps(unsigned int seg_idx, float max_overlap_allowed = 0.2) const;

  /**
   * @brief Fits a rotated rectangle to the group of segments + a new candidate.
   * @param candidate The index of the candidate line segment to be added in the group.
   * The rectangle will be fitted as if the candidate were part of the group.
   * @return The rectangle fitted to the group + candidate
   */
  cv::RotatedRect fitRectangleToCandidate(unsigned int candidate);

  /**
   * @brief Computes the overlap between two line segments, projecting candidateSeg over mainSeg.
   * @param mainSeg The segment where candidateSeg will be projected
   * @param candidateSeg The projected segment
   * @return A value between 0 and 1, where 0 means no overlap and 1 total overlap
   */
  static float segmentsOverlap(const Segment &mainSeg, const Segment &candidateSeg);

 protected:
  Segment mBaseSeg;
  double mCachedAngleMeanSin = 0;
  double mCachedAngleMeanCos = 0;
  // Fields
  std::vector<unsigned int> mSegmentsIds;
  const Segments *mAllImgSegments = nullptr;

 private:
  void recalculateLineEquation();

  void recalculateBaseSeg(const Segment &newSegment);

  void internalRecalculateBaseSeg(const Segment &new_seg);

  // Least square parameters
  double sum_x_i = 0;
  double sum_y_i = 0;
  double sum_x_i_2 = 0;
  double sum_x_i_y_i = 0;
  double N = 0;
  cv::Vec3d mLineEquation;
  bool isHorizontal = true;
};

typedef std::shared_ptr<SegmentsGroup> SegmentsGroupPtr;

void filterSegments(const Segments &originalSegs,
                    const SegmentClusters &clusters,
                    Segments &filteredSegs,  // NOLINT
                    Segments &noisySegs,  // NOLINT
                    double lenThres = 30);
}  // namespace upm

#endif  // FSG_SEGMENTSGROUP_H_
