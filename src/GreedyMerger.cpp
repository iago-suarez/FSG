#include <utility>
#include <memory>
#include "GreedyMerger.h"
#include "Utils.h"

#define UPM_RADS_TO_DEGREES 57.295779513082  // (180.0 / M_PI)

namespace upm {

GreedyMerger::GreedyMerger(cv::Size imgSize) : mImgSize(imgSize) {
  auto validatorPtr = std::make_shared<FastAcontrarioValidator>();
  validatorPtr->cachLogCombinationsForImage(imgSize);
  mValidator = validatorPtr;
}

void GreedyMerger::mergeSegments(const Segments &originalSegments,
                                 Segments &mergedLines,
                                 SegmentClusters &assignations) {
  // Common variable declaration
  int N, selected_seg_bin, theta_bin, prev_theta_bin, next_theta_bin;
  unsigned int i;
  unsigned int dst_seg_idx, prev_bin_idx, next_bin_idx;
  double theta;
  std::vector<unsigned int> M_segs, M_segs_inv;
  uint64 bin_plus_prev_n_elements;
  float l_1_x_start_ptn, l_2_x_start_ptn;
  std::pair<cv::Vec3f, cv::Vec3f> conic;
  double min_distance, endPointsDist, validityMeasure;
  int min_distance_idx, min_distance_M_idx;
  unsigned int selected_idx;
  unsigned int position_in_M;

  const double ORIENTATION_BINS_BY_RAD = N_ORIENTATION_BINS / M_PI;
  const double MIN_SEGMENT_SIZE_THRES = MIN_SEGMENT_SIZE * (mImgSize.width + mImgSize.height) / 2.0;

  // 1. Semi-Sort all the segments by its orientation in a set P_theta
  std::vector<unsigned int> L = partialSortByLength(originalSegments, N_LENGTH_BINS, mImgSize);

  // 2. Semi-Sort all the segments by its length in a set L
  std::vector<std::vector<unsigned int> > P_theta =
      getOrientationHistogram(originalSegments, N_ORIENTATION_BINS);

  std::vector<bool> is_seg_merged(originalSegments.size(), false);
  // 3. For each segment s in L (from largest to shortest)
  for (unsigned int seg_idx : L) {
    if (segLength(originalSegments[seg_idx]) < MIN_SEGMENT_SIZE_THRES) {
      // Save the output line candidate in the output arguments
      SegmentsGroupPtr base_seg = std::make_shared<SegmentsGroup>(&originalSegments);
      base_seg->addNewSegment(seg_idx);
      mergedLines.push_back(base_seg->getBaseSeg());
      assignations.push_back(base_seg->getSegmentIds());
      continue;
    }

    // 3.1 If the segment has been included as part of some group previously
    // created advance to the next segment in the L vector.
    if (is_seg_merged[seg_idx]) {
      // LOGD << "\t The segment has been already merged, so we're going to skip it";
      continue;
    }

    // 3.2 Create a segments group representation called BaseSegment,
    // that initially only contains the selected segment
    // Delete the current segment from the P_theta bin
    theta = segOrientation(originalSegments[seg_idx]);
    selected_seg_bin = (M_PI_2 + theta) * ORIENTATION_BINS_BY_RAD;
    P_theta[selected_seg_bin].erase(
        std::remove(P_theta[selected_seg_bin].begin(),
                    P_theta[selected_seg_bin].end(),
                    seg_idx),
        P_theta[selected_seg_bin].end());

    is_seg_merged[seg_idx] = true;
    SegmentsGroupPtr base_seg = std::make_shared<SegmentsGroup>(&originalSegments);
    base_seg->addNewSegment(seg_idx);

    // 3.3.1 Select the nearest segments in the same P_theta bin that s and the adjacent -> N
    theta_bin = (M_PI_2 + theta) * ORIENTATION_BINS_BY_RAD;
    prev_theta_bin = (theta_bin != 0) ? theta_bin - 1 : N_ORIENTATION_BINS - 1;
    next_theta_bin = (theta_bin != N_ORIENTATION_BINS - 1) ? theta_bin + 1 : 0;

    // 3.3.4 Create a matrix with the middle points of the segments with an
    // angle similar to that of base_seg: P_theta[theta_bin], P_theta[prev_theta_bin], P_theta[next_theta_bin]
    assert(theta_bin >= 0 && theta_bin < P_theta.size());
    assert(prev_theta_bin >= 0 && prev_theta_bin < P_theta.size());
    assert(next_theta_bin >= 0 && next_theta_bin < P_theta.size());

    N = P_theta[theta_bin].size() + P_theta[prev_theta_bin].size() + P_theta[next_theta_bin].size();

    // The M matrix contains the middlepoints of the candidate segments to be included in the base seg.
    cv::Mat M(3, N, CV_32FC1);

    // List of the segments that are inside M
    M_segs.resize(N);
    // This vector tell us for each of the original segments, in what
    // position is the middlepoint of this segment inside M
    M_segs_inv.resize(originalSegments.size());

    // Add all the segments in P_theta[theta_bin] to the M matrix
    for (i = 0; i < P_theta[theta_bin].size(); i++) {
      dst_seg_idx = P_theta[theta_bin][i];
      M_segs[i] = dst_seg_idx;
      M_segs_inv[dst_seg_idx] = i;
      M.at<float>(0, i) = (originalSegments[dst_seg_idx][0] + originalSegments[dst_seg_idx][2]) / 2.0f;
      M.at<float>(1, i) = (originalSegments[dst_seg_idx][1] + originalSegments[dst_seg_idx][3]) / 2.0f;
      M.at<float>(2, i) = -1.0f;
    }

    // Add all the segments in P_theta[prev_theta_bin] to the M matrix
    bin_plus_prev_n_elements = P_theta[theta_bin].size() + P_theta[prev_theta_bin].size();
    for (i = P_theta[theta_bin].size(); i < bin_plus_prev_n_elements; i++) {
      // The element i of the M is the element in the
      // prev_bin_idx position of P_theta[prev_theta_bin]
      prev_bin_idx = i - P_theta[theta_bin].size();
      dst_seg_idx = P_theta[prev_theta_bin][prev_bin_idx];
      M_segs[i] = dst_seg_idx;
      M_segs_inv[dst_seg_idx] = i;
      // Middle point X
      M.at<float>(0, i) = (originalSegments[dst_seg_idx][0] + originalSegments[dst_seg_idx][2]) / 2.0f;
      // Middle point Y
      M.at<float>(1, i) = (originalSegments[dst_seg_idx][1] + originalSegments[dst_seg_idx][3]) / 2.0f;
      M.at<float>(2, i) = -1.0f;
    }

    // Add all the segments in P_theta[next_theta_bin] to the M matrix
    for (i = bin_plus_prev_n_elements; i < N; i++) {
      // The element i of the M is the element in the
      // next_bin_idx position of P_theta[next_theta_bin]
      next_bin_idx = i - bin_plus_prev_n_elements;
      dst_seg_idx = P_theta[next_theta_bin][next_bin_idx];
      M_segs[i] = dst_seg_idx;
      M_segs_inv[dst_seg_idx] = i;
      // Middle point X
      M.at<float>(0, i) = (originalSegments[dst_seg_idx][0] + originalSegments[dst_seg_idx][2]) / 2.0f;
      // Middle point Y
      M.at<float>(1, i) = (originalSegments[dst_seg_idx][1] + originalSegments[dst_seg_idx][3]) / 2.0f;
      M.at<float>(2, i) = -1.0f;
    }

    // 3.0 While there are segments that can be added
    int iteration = 0;
    while (N > 0) {
      iteration++;
      //3.3.2 Based on the maximum error for each endpoint of the segment, we
      // calculate two lines l_1 and l_2 that define the conic where all the
      // line candidate segments are contained
      conic = getTangentLineEqs(base_seg->getBaseSeg(), ENDPOINT_ERROR);
      // The third component of the line should be positive if we want that
      // the points over the line have a positive distance and those above it negative
      cv::Vec3f l_1 = conic.first[2] < 0 ? -conic.first : conic.first;
      cv::Vec3f l_2 = conic.second[2] < 0 ? -conic.second : conic.second;
      // 3.3.3 When the segment is nearly vertical the condition in the
      // segments middle-points we're going to check is that the distances
      // to both lines l_1 and l_2 have the same symbol, while when the
      // segment is nearly horizontal the condition will be that the
      // distances have different symbol, we can easily know this looking
      // the symbols generated by one of the endpoints
      l_1_x_start_ptn = l_1.dot(cv::Vec3f(base_seg->getBaseSeg()[0], base_seg->getBaseSeg()[1], -1));
      l_2_x_start_ptn = l_2.dot(cv::Vec3f(base_seg->getBaseSeg()[0], base_seg->getBaseSeg()[1], -1));
      // If the endpoint distances has different sign invert one of the,
      if ((l_1_x_start_ptn > 0) != (l_2_x_start_ptn > 0))
        l_1 = -l_1;

      // 3.3.5 Do the multiplication of the created matrix M_segs_new by the normalized
      // equation of l_1 and l_2. The result will be a vector of size 1xN,
      // where, when the segment is inside the conic the sign of l_1 x s_i
      // should be different that l_2 x s_i.
      cv::Mat l1_x_M = cv::Mat(l_1.t()) * M;
      cv::Mat l2_x_M = cv::Mat(l_2.t()) * M;

      // 3.4 Create a set of matching segments M_segs_new with those segments matching this criteria
      min_distance = mValidator->getEpsilon();
      min_distance_idx = -1, min_distance_M_idx = -1;
      // M_segs_new contains the index of the segments that are inside the conic
      std::vector<unsigned int> M_segs_new;

      for (i = 0; i < N; i++) {
        // If the segment with index i is qwertyuiop the two lines that define the conic
        const bool have_same_symbol = (l1_x_M.at<float>(0, i) > 0) == (l2_x_M.at<float>(0, i) > 0);
        if (have_same_symbol) {
          // Retrieve the index of the segment we are currently looking
          selected_idx = M_segs[i];
          M_segs_new.push_back(selected_idx);

          endPointsDist = nearestEndpointDist(*base_seg, originalSegments[selected_idx]);
          if (endPointsDist > ENDPOINTS_MAX_DISTANCE) {
            continue;
          }
          validityMeasure = mValidator->candidateValidity(base_seg, originalSegments, selected_idx);
          if (validityMeasure < min_distance) {
            if (!DO_NOT_MERGE_OVERLAPPING_SEGS ||
                (DO_NOT_MERGE_OVERLAPPING_SEGS && !base_seg->overlaps(selected_idx))) {
              min_distance = validityMeasure;
              min_distance_idx = selected_idx;
              min_distance_M_idx = M_segs_new.size() - 1;
            }
          }
        }
      }

      if (min_distance_idx == -1) {
        // Breaking loop because there are no more segments to add
        break;
      }

      M_segs_new.erase(M_segs_new.begin() + min_distance_M_idx);
      N = M_segs_new.size();

      cv::Mat M_new(3, N, CV_32FC1);
      for (i = 0; i < N; i++) {
        position_in_M = M_segs_inv[M_segs_new[i]];
        M_new.at<float>(0, i) = M.at<float>(0, position_in_M);
        M_new.at<float>(1, i) = M.at<float>(1, position_in_M);
        M_new.at<float>(2, i) = -1.0f;
        // Update the M_segs_inv entry
        M_segs_inv[M_segs_new[i]] = i;
      }

      M = M_new;
      M_segs = M_segs_new;

      //Merge the nearest segment with the previous selected and start a new interaction
      base_seg->addNewSegment(min_distance_idx);
      // Remove the merged segment from the P_theta histogram
      selected_seg_bin = (M_PI_2 + segOrientation(originalSegments[min_distance_idx])) * ORIENTATION_BINS_BY_RAD;

      // Check that the array P_theta[selected_seg_bin] contains the element min_distance_idx
      assert(std::find(P_theta[selected_seg_bin].begin(),
                               P_theta[selected_seg_bin].end(),
                               min_distance_idx) != P_theta[selected_seg_bin].end());
      P_theta[selected_seg_bin].erase(
          std::remove(P_theta[selected_seg_bin].begin(),
                      P_theta[selected_seg_bin].end(),
                      min_distance_idx),
          P_theta[selected_seg_bin].end());

      // Check that the array P_theta[selected_seg_bin] doesn't contain the element min_distance_idx
      assert(std::find(P_theta[selected_seg_bin].begin(),
                               P_theta[selected_seg_bin].end(),
                               min_distance_idx) == P_theta[selected_seg_bin].end());

      is_seg_merged[min_distance_idx] = true;
    }
    // Save the output line candidate in the output arguments
    mergedLines.push_back(base_seg->getBaseSeg());
    assignations.push_back(base_seg->getSegmentIds());
  }
}

SegmentClusters
GreedyMerger::getOrientationHistogram(const Segments &segments, int bins) {
  std::vector<std::vector<unsigned int> > histogram(bins);

  const double ORIENTATION_BINS_BY_RAD = bins / M_PI;
  for (unsigned int i = 0; i < segments.size(); i++) {
    const double orientation = segOrientation(segments[i]);
    const int bin = (M_PI_2 + orientation) * ORIENTATION_BINS_BY_RAD;
    assert(bin >= 0 && bin < histogram.size());
    histogram[bin].push_back(i);
  }
  return histogram;
}

std::vector<unsigned int>
GreedyMerger::partialSortByLength(const Segments &segments, int bins, const cv::Size& imgSize) {
  std::vector<unsigned int> result;
  std::vector<std::vector<unsigned int> > histogram(bins);

  // The MAX_VAL is the diagonal
  const double MAX_VAL = sqrt(imgSize.width * imgSize.width + imgSize.height * imgSize.height);
  const double BIN_AMOUNT = MAX_VAL / static_cast<double>(bins);
  for (unsigned int i = 0; i < segments.size(); i++) {
    const double length = segLength(segments[i]);
    auto bin = static_cast<int>(length / BIN_AMOUNT);

    if (!(bin >= 0 && bin < histogram.size())) {
      std::cerr << "There is a segment longer than the diagonal! "
              "Have you initialize the GreedyMerger with the image size? used: " << imgSize << std::endl;
      throw std::invalid_argument("GreedyMerger: There is a segment larger than the diagonal!");
    }

    histogram[bin].push_back(i);
  }
  for (int h_col = bins - 1; h_col >= 0; h_col--) {
    result.insert(std::end(result), histogram[h_col].begin(), histogram[h_col].end());
  }

  return result;
}

void GreedyMerger::setImageSize(const cv::Size &size) {
  mImgSize = size;
  auto validator = std::dynamic_pointer_cast<FastAcontrarioValidator>(mValidator);
  if (validator) {
    validator->cachLogCombinationsForImage(size);
  }
}

std::pair<cv::Vec3f, cv::Vec3f>
GreedyMerger::getTangentLineEqs(Segment seg, float r) {
  const cv::Point2d endPoint(seg[0], seg[1]);
  const cv::Point2d middlePoint((seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2);

  const cv::Point2d p1ic = middlePoint - endPoint;
  const double norm_p1ic = cv::norm(p1ic);
  const double d = sqrt(norm_p1ic * norm_p1ic - r * r);
  const double pap1_dist = r * r / norm_p1ic;
  const double papb_dist = r * d / norm_p1ic;
  const cv::Point2d p1pi_dir = (p1ic / norm_p1ic);
  const cv::Point2d pa = p1pi_dir * pap1_dist + endPoint;
  const cv::Point2d pb_first = cv::Point2d(-p1pi_dir.y, p1pi_dir.x) * papb_dist + pa;
  const cv::Point2d pb_second = cv::Point2d(p1pi_dir.y, -p1pi_dir.x) * papb_dist + pa;

  cv::Vec3f l1 = cv::Vec3d(pb_first.x, pb_first.y, -1).cross(cv::Vec3d(middlePoint.x, middlePoint.y, -1));
  l1 /= std::sqrt(l1[0] * l1[0] + l1[1] * l1[1]);
  // Since the polar equation is x*cos(theta) + y*sin(theta) = rho
  // And theta is defined in the range [-pi/2 ; pi/2), cos(theta) will never
  // be a negative number, and so, we should invert the equation symbol if it's negative
  if (l1[0] < 0) {
    // If the value is negative change the symbol of the equation
    l1 *= -1;
  }

  cv::Vec3f l2 = cv::Vec3d(pb_second.x, pb_second.y, -1).cross(cv::Vec3d(middlePoint.x, middlePoint.y, -1));
  l2 /= std::sqrt(l2[0] * l2[0] + l2[1] * l2[1]);
  if (l2[0] < 0) l2 *= -1;

  return {l1, l2};
}

double
GreedyMerger::nearestEndpointDist(SegmentsGroup &base_seg, const Segment &seg) {
  const Segment &baseSeg = base_seg.getBaseSeg();
  double basSegLength = segLength(baseSeg);
  float dx, dy, currDist;
  dx = baseSeg[0] - seg[0];
  dy = baseSeg[1] - seg[1];
  currDist = std::sqrt(dx * dx + dy * dy);
  float minDistPx = currDist;
  float maxDistPx = currDist;
  dx = baseSeg[0] - seg[2];
  dy = baseSeg[1] - seg[3];
  currDist = std::sqrt(dx * dx + dy * dy);
  minDistPx = std::min(minDistPx, currDist);
  maxDistPx = std::max(maxDistPx, currDist);

  dx = baseSeg[2] - seg[0];
  dy = baseSeg[3] - seg[1];
  currDist = std::sqrt(dx * dx + dy * dy);
  minDistPx = std::min(minDistPx, currDist);
  maxDistPx = std::max(maxDistPx, currDist);

  dx = baseSeg[2] - seg[2];
  dy = baseSeg[3] - seg[3];
  currDist = std::sqrt(dx * dx + dy * dy);
  minDistPx = std::min(minDistPx, currDist);
  maxDistPx = std::max(maxDistPx, currDist);

  if (maxDistPx < basSegLength) {
    return 0;
  }
  double minLen = 0.5 * (segLength(seg) + basSegLength / (float) base_seg.getSegmentIds().size());
  return minDistPx / minLen;
}

void GreedyMerger::setNumOrientationBins(int numOrientationBins) {
  N_ORIENTATION_BINS = numOrientationBins;
}

void GreedyMerger::setNumLengthBins(int numLengthBins) {
  N_LENGTH_BINS = numLengthBins;
}

void GreedyMerger::setEndPointError(float endpointError) {
  ENDPOINT_ERROR = endpointError;
}

void GreedyMerger::setMinSegSize(double minSegSize) {
  MIN_SEGMENT_SIZE = minSegSize;
}

void GreedyMerger::setEndpointsMaxDistance(double maxDistance) {
  ENDPOINTS_MAX_DISTANCE = maxDistance;
}

void GreedyMerger::setSegmentsValidator(std::shared_ptr<FastAcontrarioValidator> validator)  {
  mValidator = std::move(validator);
}

}