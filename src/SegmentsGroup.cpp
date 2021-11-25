#include "SegmentsGroup.h"
#include "Utils.h"

#define UPM_RADS_TO_DEGREES 57.295779513082  // (180.0 / M_PI)
#define UPM_ABS(x) ((x) >= 0 ? (x) : -(x))

namespace upm {

cv::Vec4d
SegmentsGroup::getBaseSeg() const {
  return mBaseSeg;
}

bool
SegmentsGroup::overlaps(unsigned int seg_idx, float max_overlap_allowed) const {
  float amount_of_overlap = 0;
  for (unsigned int els_idx : mSegmentsIds) {
    amount_of_overlap += segmentsOverlap((*mAllImgSegments)[els_idx], (*mAllImgSegments)[seg_idx]);
    if (amount_of_overlap > max_overlap_allowed) {
      return true;
    }
  }
  return false;
}

inline bool same_dir(const cv::Point2f &v1, const cv::Point2f &v2) {
  return v1.dot(v2) >= 0;
}

float
SegmentsGroup::segmentsOverlap(const Segment &mainSeg, const Segment &candidateSeg) {
  // Calculate the equation of the line passing by the main segment
  cv::Vec3f l = cv::Vec3f(mainSeg[0], mainSeg[1], 1).cross(cv::Vec3f(mainSeg[2], mainSeg[3], 1));

  const cv::Point2f &startProj = getProjectionPtn(l, cv::Point2f(candidateSeg[0], candidateSeg[1]));
  const cv::Point2f &endProj = getProjectionPtn(l, cv::Point2f(candidateSeg[2], candidateSeg[3]));

  // This is a vector starting in the Start endpoint and finishing on the End endpoint of the main segment
  const cv::Point2f mainStart(mainSeg[0], mainSeg[1]);
  const cv::Point2f &vMain = cv::Point2f(mainSeg[2], mainSeg[3]) - mainStart;
  // Those are vectors from the start point of the main segment to the
  // projections of the endpoints of the candidate segment
  const cv::Point2f &vProjStart = startProj - mainStart;
  const cv::Point2f &vProjEnd = endProj - mainStart;

  const double normVMain = cv::norm(vMain), normProjStart = cv::norm(vProjStart), normProjEnd = cv::norm(vProjEnd);
  // Check some basic conditions
  const bool same_dir_main_start = same_dir(vMain, vProjStart);
  const bool same_dir_main_end = same_dir(vMain, vProjEnd);
  const bool startOverlap = (same_dir_main_start && normProjStart <= normVMain);
  const bool endOverlap = (same_dir_main_end && normProjEnd <= normVMain);

  // Now we're going to check what the position of the two segments that can be:
  // 1. The endpoints doesn't overlap:
  //    1.1. Because the candidateSeg is larger than the mainSeg. In this case the segments totally overlap!
  //    1.2. Because the segments are not overlapped
  // 2. Both endpoints overlap, so the segments are totally overlapped.
  // 3. The segments are partially overlapped:
  //    3.1. They overlap by the mainSegments start point
  //    3.2. They overlap by the mainSegments end point
  if (!startOverlap && !endOverlap) {
    // Check if the endpoints don't overlap because the candidate segment is bigger than the main one
    if ((same_dir_main_end && normVMain <= normProjEnd && !same_dir_main_start) ||
        (same_dir_main_start && normVMain <= normProjStart && !same_dir_main_end)) {
      // Case 1.1
      return 1;
    } else {
      // If the fist condition is not true is because th segments do not overlap at all.
      // Case 1.2
      return 0;
    }
  }
  if (startOverlap && endOverlap) {
    // Case 2
    return 1;
  }
  // One endpoint is in and the other out
  double mayorNorm, minorNorm;

  cv::Point2f mayorVProj, minorVProj;
  if (vMain.dot(vProjStart) > vMain.dot(vProjEnd)) {
    mayorVProj = vProjStart;
    minorVProj = vProjEnd;
    mayorNorm = normProjStart;
    minorNorm = normProjEnd;
  } else {
    mayorVProj = vProjEnd;
    minorVProj = vProjStart;
    mayorNorm = normProjEnd;
    minorNorm = normProjStart;
  }

  if (vMain.dot(minorVProj) < 0) {
    // minorVProj is negative and mayorVProj is positive
    assert(vMain.dot(mayorVProj) >= 0);
    // Case 3.1
    return mayorNorm / (mayorNorm + minorNorm);
  } else {
    // Both minorVProj and mayorVProj are positives
    // Case 3.2
    return (normVMain - minorNorm) / (mayorNorm - minorNorm);
  }
}

const std::vector<unsigned int> &
SegmentsGroup::getSegmentIds() const {
  return mSegmentsIds;
}

cv::RotatedRect SegmentsGroup::fitRectangleToCandidate(unsigned int candidateIdx) {
  // Calculate the angle as the weathered mean of the rectangle angle.
  // The angle mean is a circular mean, see: https://en.wikipedia.org/wiki/Mean_of_circular_quantities
  double cachedAngleMeanSin = mCachedAngleMeanSin;
  double cachedAngleMeanCos = mCachedAngleMeanCos;

  double angle;

  // Compute the rectangle orientation as the mean of the segment
  // orientations weighted by the square root of its length

  ////////////////////////////////////////////////////////////////////
  double dx = (*mAllImgSegments)[candidateIdx][0] - (*mAllImgSegments)[candidateIdx][2];
  double dy = (*mAllImgSegments)[candidateIdx][1] - (*mAllImgSegments)[candidateIdx][3];
  // The weight is the square root of the length of the segment
  const double factor = std::pow(dx * dx + dy * dy, -0.25);
  // Ensure that the angle is between -pi/2 and pi/2
  if (dx < 0) {
    dx = -dx;
    dy = -dy;
  }
  cachedAngleMeanCos += factor * dx;
  cachedAngleMeanSin += factor * dy;
  ////////////////////////////////////////////////////////////////////

  angle = std::atan2(cachedAngleMeanSin, cachedAngleMeanCos);

  // Rotate the points
  // now apply rotation
  double cos_angle = cos(angle);
  double sin_angle = sin(angle);

  double max_rot_x = -DBL_MAX, min_rot_x = DBL_MAX, max_rot_y = -DBL_MAX, min_rot_y = DBL_MAX;

  cv::Mat R = cv::Mat(cv::Matx22d(cos_angle, -sin_angle,
                                  sin_angle, cos_angle));
  for (unsigned int element : mSegmentsIds) {
    const Segment &s = (*mAllImgSegments)[element];

    cv::Point2d rotStart(+cos_angle * s[0] + sin_angle * s[1],
                         -sin_angle * s[0] + cos_angle * s[1]);
    cv::Point2d rotEnd(+cos_angle * s[2] + sin_angle * s[3],
                       -sin_angle * s[2] + cos_angle * s[3]);

    if (rotStart.x > max_rot_x) max_rot_x = rotStart.x;
    if (rotStart.x < min_rot_x) min_rot_x = rotStart.x;
    if (rotStart.y > max_rot_y) max_rot_y = rotStart.y;
    if (rotStart.y < min_rot_y) min_rot_y = rotStart.y;

    if (rotEnd.x > max_rot_x) max_rot_x = rotEnd.x;
    if (rotEnd.x < min_rot_x) min_rot_x = rotEnd.x;
    if (rotEnd.y > max_rot_y) max_rot_y = rotEnd.y;
    if (rotEnd.y < min_rot_y) min_rot_y = rotEnd.y;
  }
  ////////////////////////////////////////////////////////////////////
  const Segment &mySeg = (*mAllImgSegments)[candidateIdx];

  cv::Point2d rotStart(+cos_angle * mySeg[0] + sin_angle * mySeg[1],
                       -sin_angle * mySeg[0] + cos_angle * mySeg[1]);
  cv::Point2d rotEnd(+cos_angle * mySeg[2] + sin_angle * mySeg[3],
                     -sin_angle * mySeg[2] + cos_angle * mySeg[3]);

  if (rotStart.x > max_rot_x) max_rot_x = rotStart.x;
  if (rotStart.x < min_rot_x) min_rot_x = rotStart.x;
  if (rotStart.y > max_rot_y) max_rot_y = rotStart.y;
  if (rotStart.y < min_rot_y) min_rot_y = rotStart.y;

  if (rotEnd.x > max_rot_x) max_rot_x = rotEnd.x;
  if (rotEnd.x < min_rot_x) min_rot_x = rotEnd.x;
  if (rotEnd.y > max_rot_y) max_rot_y = rotEnd.y;
  if (rotEnd.y < min_rot_y) min_rot_y = rotEnd.y;
  ////////////////////////////////////////////////////////////////////


  // Calculate the middle point with the rotated space
  double rot_middle_x = (max_rot_x + min_rot_x) / 2.0;
  double rot_middle_y = (max_rot_y + min_rot_y) / 2.0;
  // Invert the applied rotation
  double middle_x = rot_middle_x * cos_angle - rot_middle_y * sin_angle;
  double middle_y = rot_middle_x * sin_angle + rot_middle_y * cos_angle;

  // Calculate the length and the width of the rectangle
  double rectangle_len = max_rot_x - min_rot_x;
  double rectangle_width = max_rot_y - min_rot_y;

  double angleDegrees = 90 + angle * UPM_RADS_TO_DEGREES;
  return cv::RotatedRect(cv::Point2f(middle_x, middle_y), cv::Size2f(rectangle_width, rectangle_len), angleDegrees);
}

cv::Vec3d SegmentsGroup::getLineEquation() const {
  return mLineEquation;
}

void SegmentsGroup::recalculateLineEquation() {
  // The homogeneous line equation in format ax + by + c = 0
  cv::Vec3d l;

  // Since:
  // ax + by + c = 0
  // y = mx + n
  // m = a / -b   ,   n = c / -b
  // We avoid the calculation of m and n that is not valid for vertical lines
  const double a = N * sum_x_i_y_i - sum_x_i * sum_y_i;
  const double b = -(N * sum_x_i_2 - sum_x_i * sum_x_i);
  const double c = sum_y_i * sum_x_i_2 - sum_x_i * sum_x_i_y_i;
  l = isHorizontal ? cv::Vec3d(a, b, c) : cv::Vec3d(b, a, c);

  // Normalize the line
  l /= std::sqrt(l[0] * l[0] + l[1] * l[1]);

  // Since the polar equation is x*cos(theta) + y*sin(theta) = rho
  // And theta is defined in the range [-pi/2 ; pi/2), cos(theta) will never
  // be a negative number, and so, we should invert the equation symbol if it's negative
  if (l[0] < 0) {
    // If the distance is negative change the symbol of the equation
    l = -l;
  }

  mLineEquation = l;
}

void SegmentsGroup::internalRecalculateBaseSeg(const Segment &new_seg) {
  if (mSegmentsIds.size() == 1) {
    // If this is the first point
    mBaseSeg = new_seg;
    return;
  }

  // The homogeneous line equation in format ax + by + c = 0
  cv::Vec3d l = getLineEquation();

  const cv::Point2f new_start_ptn(new_seg[0], new_seg[1]);
  const cv::Point2f new_end_ptn(new_seg[2], new_seg[3]);

  // Now select the far segment endpoints
  cv::Point2d currStartPoint, currEndPoint, startPoint, endPoint;
  double dist, max_dist;

  // base_start - base_end
  currStartPoint = startPoint = cv::Point2d(mBaseSeg[0], mBaseSeg[1]);
  currEndPoint = endPoint = cv::Point2d(mBaseSeg[2], mBaseSeg[3]);
  dist = cv::norm(currEndPoint - currStartPoint);
  max_dist = dist;

  // base_start - new_start
  currStartPoint = cv::Point2d(mBaseSeg[0], mBaseSeg[1]);
  currEndPoint = new_start_ptn;
  dist = cv::norm(currEndPoint - currStartPoint);
  if (dist > max_dist) {
    max_dist = dist;
    startPoint = currStartPoint;
    endPoint = currEndPoint;
  }

  // base_start - new_end
  currStartPoint = cv::Point2d(mBaseSeg[0], mBaseSeg[1]);
  currEndPoint = new_end_ptn;
  dist = cv::norm(currEndPoint - currStartPoint);
  if (dist > max_dist) {
    max_dist = dist;
    startPoint = currStartPoint;
    endPoint = currEndPoint;
  }

  // base_end - new_start
  currStartPoint = cv::Point2d(mBaseSeg[2], mBaseSeg[3]);
  currEndPoint = new_start_ptn;
  dist = cv::norm(currEndPoint - currStartPoint);
  if (dist > max_dist) {
    max_dist = dist;
    startPoint = currStartPoint;
    endPoint = currEndPoint;
  }
  // base_end - new_end
  currStartPoint = cv::Point2d(mBaseSeg[2], mBaseSeg[3]);
  currEndPoint = new_end_ptn;
  dist = cv::norm(currEndPoint - currStartPoint);
  if (dist > max_dist) {
    startPoint = currStartPoint;
    endPoint = currEndPoint;
  }

  // Project both endpoint into the line equation
  const cv::Point2d &projStartP = getProjectionPtn(l, startPoint);
  const cv::Point2d &projEndP = getProjectionPtn(l, endPoint);
  mBaseSeg = cv::Vec4d(projStartP.x, projStartP.y, projEndP.x, projEndP.y);
}


void SegmentsGroup::recalculateBaseSeg(const Segment &newSegment) {
  if (mSegmentsIds.size() == 1) {
    const double dx = UPM_ABS(newSegment[0] - newSegment[2]);
    const double dy = UPM_ABS(newSegment[1] - newSegment[3]);
    isHorizontal = dx >= dy;
  }
  const cv::Point2f new_start_ptn(isHorizontal ? newSegment[0] : newSegment[1],
                                  isHorizontal ? newSegment[1] : newSegment[0]);
  const cv::Point2f new_end_ptn(isHorizontal ? newSegment[2] : newSegment[3],
                                isHorizontal ? newSegment[3] : newSegment[2]);

  sum_x_i += new_start_ptn.x + new_end_ptn.x;
  sum_y_i += new_start_ptn.y + new_end_ptn.y;
  sum_x_i_2 += new_start_ptn.x * new_start_ptn.x + new_end_ptn.x * new_end_ptn.x;
  sum_x_i_y_i += new_start_ptn.x * new_start_ptn.y + new_end_ptn.x * new_end_ptn.y;
  N += 2;

  recalculateLineEquation();

  // Call to parent, that will call to getLineEquation method in the child again
  internalRecalculateBaseSeg(newSegment);
}

void SegmentsGroup::addNewSegment(unsigned int newSegmentId) {
  mSegmentsIds.push_back(newSegmentId);
  recalculateBaseSeg((*mAllImgSegments)[newSegmentId]);
}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

void filterSegments(const Segments &originalSegs,
                    const SegmentClusters &clusters,
                    Segments &filteredSegs,
                    Segments &noisySegs,
                    double lenThres) {
  for (auto &cluster : clusters) {
    if (cluster.size() == 1) {
      if (segLength(originalSegs[cluster[0]]) < lenThres) {
        noisySegs.push_back(originalSegs[cluster[0]]);
      } else {
        filteredSegs.push_back(originalSegs[cluster[0]]);
      }
    } else {
      SegmentsGroup base_seg(&originalSegs);
      for (unsigned int i : cluster)
        base_seg.addNewSegment(i);
      filteredSegs.push_back(base_seg.getBaseSeg());
    }
  }

}
}