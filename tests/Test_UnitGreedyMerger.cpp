/**
 * @copyright 2018 Xoan Iago Suarez Canosa. All rights reserved.
 * Constact: iago.suarez.canosa@alumnos.upm.es
 * Software developed in the PhD: Augmented Reality for Urban Environments
 */
#include "GreedyMerger.h"
//#include "TestingComparators.h"
#include "SegmentsGroup.h"
#include "SegmentClusteringXmlParser.h"
#include <gtest/gtest.h>
#include <LsdOpenCV.h>
//#include "MergerEvaluation.h"
//#include "SegmentClusteringXmlParser.h"

#define UPM_PTS_ERROR_THRES 2 // In pixels

//#define TELEFONICA_IMAGE upm::getResource("/datasets/MadridBuildings/escorial/original.jpg")
#define TELEFONICA_IMAGE "/home/iago/workspace/line-experiments/resources/datasets/MadridBuildings/telefonica/original.jpg"
//#define TELEFONICA_IMAGE upm::getResource("/windows.jpg")

using namespace upm;

double
calcJaccardIdx(const std::vector<unsigned int> &C1, const std::vector<unsigned int> &C2) {
  int nIntersectionElements = 0;
  for (unsigned int e1 : C1) {
    for (unsigned int e2 : C2) {
      if (e1 == e2) nIntersectionElements++;
    }
  }
  return nIntersectionElements / (double) (C1.size() + C2.size() - nIntersectionElements);
}


double
jaccardIndex(const SegmentClusters &groundTruthClusts,
             const SegmentClusters &detectedClusts,
             double minJaccard = 0.25) {
  double sumOfJaccards = 0;
  double nSums = 0;
  for (const std::vector<unsigned int> &gtClus : groundTruthClusts) {
    for (const std::vector<unsigned int> &detClus : detectedClusts) {
      double jaccardVal = calcJaccardIdx(gtClus, detClus);
      if (jaccardVal > minJaccard) {
        sumOfJaccards += jaccardVal;
        nSums++;
      }
    }
  }
  if (nSums == 0) return 0;
  return sumOfJaccards / nSums;
}

bool containsSeg(const Segment &obtained_seg, const Segments &expectedVals) {

  bool found = false;
  cv::Point2f obtained_start(obtained_seg[0], obtained_seg[1]);
  cv::Point2f obtained_end(obtained_seg[2], obtained_seg[3]);
  for (auto &val : expectedVals) {
    cv::Point2f start(val[0], val[1]);
    cv::Point2f end(val[2], val[3]);
    const bool found_start =
        cv::norm(obtained_start - start) < UPM_PTS_ERROR_THRES || cv::norm(obtained_start - end) < UPM_PTS_ERROR_THRES;
    const bool found_end =
        cv::norm(obtained_end - end) < UPM_PTS_ERROR_THRES || cv::norm(obtained_end - start) < UPM_PTS_ERROR_THRES;
    found = found_start & found_end;
    if (found) break;
  }
  return found;
}

cv::Vec3f
get_line_equation(cv::Mat F_b) {
  cv::Vec3f res;
  cv::Mat w, u, vt;
  cv::SVD::compute(F_b, w, u, vt, cv::SVD::FULL_UV);
  // F_b(3x3) = U(3x3) * V^t (3x3)
  // Copy the eigenvector associated with the smallest eigenvalue
  for (int i = 0; i < 3; i++) {
    switch (vt.type()) {
      case CV_64FC1:res[i] = vt.at<double>(2, i);
        break;
      case CV_32FC1:res[i] = vt.at<float>(2, i);
        break;
      default:break;
    }
  }
  // Normalize the line
  res /= std::sqrt(res[0] * res[0] + res[1] * res[1]);
  // Since the polar equation is x*cos(theta) + y*sin(theta) = rho
  // And theta is defined in the range [-pi/2 ; pi/2), cos(theta) will never
  // be a negative number, and so, we should invert the equation symbol if it's negative
  if (res[0] < 0) {
    // If the distance is negative change the symbol of the equation
    res *= -1;
  }
  return res;
}


enum SegmentMergerRecallOfType {
  UPM_RECALL = INT_MAX,
  UPM_PRECISION = 0,
  UPM_F_SCORE = 3
};

/**
 * @brief This returns the recall in all the image clusters of the F-score
 * inside each cluster.
 *
 * To do it, first this function find existing clusters in both the detected
 * result and the ground truth, we can do it, considering that one cluster
 * is the same if more than half of its segments in both clusters are the
 * same, i.e. the recall > 0.5.
 *
 * Once that we have a match between cluster, we evaluate each cluster with
 * the F-Score metric, and finally, since to evaluate the behaviour of the
 * clustering algorithm over all the image, not only over each cluster, we
 * return the recall where the True Positives are the F-score values for
 * each matched cluster.
 *
 * @param groundtruthClusts The hand labeled clusters
 * @param detectedClusts The detected clusters using an segment merger algorithm
 * @param goodClusters A tuple of 3 elements with the matched clusters and
 * its F-Score in the following way:
 * <index_of_ground_truth_matched_cluster, index_of_detected_matched_cluster, associated_fscore>
 * @param fScoreAlpha Determine the alpha parameter of the F-score
 * @return the recall in all the image clusters of the F-score
 * inside each cluster.
 */
double
recallOfFScores(const SegmentClusters &groundtruthClusts,
                const SegmentClusters &detectedClusts,
                std::vector<std::tuple<unsigned int, unsigned int, double> > &goodClusters,
                SegmentMergerRecallOfType fScoreAlpha = UPM_F_SCORE) {
  double sum_of_f_scores = 0;
  for (int i = 0; i < groundtruthClusts.size(); i++) {
    const std::vector<unsigned int> &gtClus = groundtruthClusts[i];
    for (int j = 0; j < detectedClusts.size(); j++) {
      const std::vector<unsigned int> &detClus = detectedClusts[j];

      int truePositive = 0, falseNegative = 0, falsePositive;
      for (unsigned int gt_seg : gtClus) {
        if (std::find(detClus.begin(), detClus.end(), gt_seg) != detClus.end()) {
          truePositive++;
        } else {
          falseNegative++;
        }
      }
      falsePositive = detClus.size() - truePositive;

      // Calculate precision and recall
      double precision = truePositive / (double) (truePositive + falsePositive);
      double recall = truePositive / (double) (truePositive + falseNegative);

      // If the recall is less than 0.5 we  consider that the clusters
      // detClus and gtClus are not the same and thus, we dont take it into
      // account
      if (recall <= 0.5)
        continue;

      // Calculate the F-score
      double alpha = fScoreAlpha; // If this is one, the F-score is the harmonic mean
      double F_alpha = (1 + alpha) / (1 / precision + alpha / recall);
      sum_of_f_scores += F_alpha;
      goodClusters.emplace_back(i, j, F_alpha);

    }
  }
  double total_recall = sum_of_f_scores / (double) groundtruthClusts.size();
  return total_recall;
}


//////////////////////////////// TESTS ////////////////////////////////

TEST(UnitGreedyMerger, SortByLength) {
  Segments ELSs = {
      Segment(10, 0, 20, 0), // Len: 10
      Segment(30, 0, 50, 0), // Len: 20
      Segment(20, -10, 20, -5), // Len 5
      Segment(500, 500, 600, 600), // Len 141,42
  };

  auto result = GreedyMerger::partialSortByLength(ELSs, 1000, cv::Size(500, 600));
  ASSERT_EQ(4, result.size());
  ASSERT_EQ(3, result[0]);
  ASSERT_EQ(1, result[1]);
  ASSERT_EQ(0, result[2]);
  ASSERT_EQ(2, result[3]);
}

template<class T>
inline std::vector<T> &
operator+=(std::vector<T> &lhs, std::vector<T> l) {
  lhs.insert(std::end(lhs), std::begin(l), std::end(l));
  return lhs;
}

TEST(UnitGreedyMerger, SortByAngle) {
  GreedyMerger merger(cv::Size(500, 600));
  Segments ELSs = {
      Segment(10, 0, 20, 0), // Theta: -1,5707
      Segment(20, 0, 10, 2), // Theta: 1,3734
      Segment(20, 0, 10, 8), // Theta: 0,8960
      // Same segment with different direction
      Segment(10, 10, 100, 100), // Theta: -0,7853
      Segment(100, 100, 10, 10), // Theta: -0,7853
      // Vertical segments
      Segment(200, 600, 200, 300), // Theta: 0
      Segment(0, 100, 0, 105), // Theta: 0
  };

  auto histogram = merger.getOrientationHistogram(ELSs, 180);

  // Put the histogram elements in a simple list sorted by orientation
  std::vector<unsigned int> result;
  for (int h_col = 0; h_col < 180; h_col++) {
    result += histogram[h_col];
  }
  ASSERT_EQ(7, result.size());
  ASSERT_EQ(0, result[0]);
  ASSERT_TRUE(result[1] == 3 || result[1] == 4);
  ASSERT_TRUE(result[2] == 3 || result[2] == 4);
  ASSERT_TRUE(result[3] == 5 || result[3] == 6);
  ASSERT_TRUE(result[4] == 5 || result[4] == 6);
  ASSERT_EQ(2, result[5]);
  ASSERT_EQ(1, result[6]);
}

TEST(UnitGreedyMerger, GetTangentLines) {
  GreedyMerger merger(cv::Size(500, 500));
  auto res = merger.getTangentLineEqs(Segment(400, 400, 400, 500), 10);
  const double NEEDED_PRECISION = 0.000001;
  ASSERT_NEAR(0.979795814, res.first[0], NEEDED_PRECISION);
  ASSERT_NEAR(-0.200000003, res.first[1], NEEDED_PRECISION);
  ASSERT_NEAR(301.918365, res.first[2], NEEDED_PRECISION);
  ASSERT_NEAR(0.979795814, res.second[0], NEEDED_PRECISION);
  ASSERT_NEAR(0.200000003, res.second[1], NEEDED_PRECISION);
  ASSERT_NEAR(481.918335, res.second[2], NEEDED_PRECISION);

  res = merger.getTangentLineEqs(Segment(200, 200, 300, 300), 5);
  ASSERT_NEAR(0.655336857, res.first[0], NEEDED_PRECISION);
  ASSERT_NEAR(-0.755336821, res.first[1], NEEDED_PRECISION);
  ASSERT_NEAR(-25.0000019, res.first[2], NEEDED_PRECISION);
  ASSERT_NEAR(0.755336821, res.second[0], NEEDED_PRECISION);
  ASSERT_NEAR(-0.655336857, res.second[1], NEEDED_PRECISION);
  ASSERT_NEAR(25.0000019, res.second[2], NEEDED_PRECISION);
}

//TODO Solve the problems with the greedy aspect
//TEST(UnitGreedyMerger, MergeSegmentsHierarchicalSyntheticMultiassignation)
//{
//  Segments ELSs = {
//      Segment(100, 100, 250, 100),
//      Segment(300, 100, 400, 100),
//      Segment(300, 102, 400, 102),
//      Segment(300, 200, 400, 200),
//      Segment(300, 120, 400, 120),
//      Segment(600, 120, 700, 120),
//      Segment(600, 150, 700, 150),
//      Segment(10, 20, 40, 10),
//      Segment(50, 50, 50, 150),
//
//      Segment(52, 200, 52, 300)
//
//  };
//  Segments mergedSegments;
//  SegmentClusters assignations;
//
//  GreedyMerger merger(cv::Size(800, 400));
//  merger.mergeSegments(ELSs, mergedSegments, assignations);
//
//  // Check that the expected segments are in the obtained
//  Segments expectedMergedSegs = {
//      Segment(100, 101, 400, 101),
//      Segment(300, 200, 400, 200),
//      Segment(300, 120, 700, 120),
//      Segment(600, 150, 700, 150),
//      Segment(10, 20, 40, 10),
//      Segment(51, 50, 51, 300)
//  };
//  ASSERT_EQ(mergedSegments.size(), expectedMergedSegs.size());
//
//  for (Segment &obtainedSeg : mergedSegments) {
//    bool contained = containsSeg(obtainedSeg, expectedMergedSegs);
//    if (!contained) {
//      std::cerr << "ERROR: Segment " << obtainedSeg
//           << " not contained in expected segments:\n" << expectedMergedSegs
//           << "\n Under tolerance: " << UPM_PTS_ERROR_THRES;
//    }
//    ASSERT_TRUE(containsSeg(obtainedSeg, expectedMergedSegs));
//  }
//}

TEST(UnitGreedyMerger, MergeSegmentsHierarchicalSynthetic) {
  Segments ELSs = {
      Segment(100, 100, 250, 100),
      Segment(300, 100, 400, 100),
      Segment(300, 102, 400, 102),
      Segment(300, 250, 400, 250),
      Segment(300, 170, 400, 170),
      Segment(600, 170, 700, 170),
      Segment(600, 200, 700, 200),
      Segment(10, 20, 40, 10),
      Segment(50, 50, 50, 150),

      Segment(52, 200, 52, 300)

  };
  Segments mergedSegments;
  SegmentClusters assignations;

  GreedyMerger merger(cv::Size(800, 400));
  merger.mergeSegments(ELSs, mergedSegments, assignations);

  // Check that the expected segments are in the obtained
  Segments expectedMergedSegs = {
      Segment(100, 100, 400, 100),
      Segment(300, 102, 400, 102),
      Segment(300, 250, 400, 250),
      Segment(300, 170, 700, 170),
      Segment(600, 200, 700, 200),
      Segment(10, 20, 40, 10),
      Segment(51, 50, 51, 300)
  };
  ASSERT_EQ(mergedSegments.size(), expectedMergedSegs.size());

  for (Segment &obtainedSeg : mergedSegments) {
    bool contained = containsSeg(obtainedSeg, expectedMergedSegs);
    if (!contained) {
      std::cerr << "ERROR: Segment " << obtainedSeg
           << " not contained in expected segments:\n" //<< expectedMergedSegs
           << "\n Under tolerance: " << UPM_PTS_ERROR_THRES << std::endl;
    }
    ASSERT_TRUE(containsSeg(obtainedSeg, expectedMergedSegs));
  }
}

TEST(UnitGreedyMerger, CheckJaccardIndex) {
  SegmentClusters C1;
  SegmentClusters C2;

  // Simple checks
  C1 = {{1, 2, 3}};
  C2 = {{1, 2, 3}};
  ASSERT_EQ(1, jaccardIndex(C1, C2));

  C1 = {{1, 2, 3}};
  C2 = {{4, 5, 6}};
  ASSERT_EQ(0, jaccardIndex(C1, C2));

  C1 = {{0, 1, 2, 3}};
  C2 = {{4, 3, 2, 1}};
  ASSERT_EQ(3 / 5.0, jaccardIndex(C1, C2));

  // With several clusters
  C1 = {
      {0, 1, 2, 3},
      {5, 6, 7},
      {9, 10, 11}
  };
  C2 = {
      {5, 6}, // With the second, 2/3
      {4, 3, 2, 1} // With the first of C1, idx: 3/5
  };

  // Expected ((10 + 9) / 15) / 2
  ASSERT_EQ(19 / 30.0, jaccardIndex(C1, C2));

  // With several clusters and one of then not valid
  C1 = {
      {0, 1, 2, 3},
      {5, 6, 7},
      {9, 10, 11}
  };
  C2 = {
      {5, 6}, // With the second, 2/3
      {4, 3, 2, 1}, // With the first of C1, idx: 3/5
      {10} // With last, idx: 1/3
  };

  // Expected
  ASSERT_EQ(8.0 / 15.0, jaccardIndex(C1, C2));
}

TEST(UnitGreedyMerger, MergeSegmentsHierarchicalReal) {
  cv::Mat image = cv::imread(TELEFONICA_IMAGE, cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    std::cerr << "can not open " << TELEFONICA_IMAGE << std::endl;
    throw std::invalid_argument("Cannot open input image");
  }

  GreedyMerger merger(image.size());
  // Minor threshold is less exigent policy
  auto validator = std::make_shared<FastAcontrarioValidator>();
  validator->setDomainSize(image.size());
  validator->setMinSegmentsDensity(1 / 500.0);
  merger.setSegmentsValidator(validator);

  Segments groupedSegments;
  SegmentClusters groups;
  Segments segments;

  LsdOpenCV detector(cv::LSD_REFINE_ADV);
  segments = detector.detect(image);
  merger.mergeSegments(segments, groupedSegments, groups);

  #ifdef _DEBUG_GTK
  //  cv::cvtColor(image, TimeTracker::instance().shared_img, CV_GRAY2BGR);

    cv::Mat copy;
    cv::cvtColor(image, copy, cv::COLOR_GRAY2BGR);
    detector->drawSegments(copy, segments);

    cv::imshow("Detected Segments", copy, 800);

    for (std::vector<unsigned int> &group : groups) {
      if (group.size() > 1) {
        cv::Scalar color(
            255.0 * rand() / (double) RAND_MAX,
            255.0 * rand() / (double) RAND_MAX,
            255.0 * rand() / (double) RAND_MAX);
        for (unsigned int idx : group) {
          auto v = segments[idx];
          cv::line(copy, cv::Point(v[0], v[1]), cv::Point(v[2], v[3]), color, 2, cv::LINE_AA);

        }
      }
    }
    cv::imshow("Grouped segments", copy, 800);
    cv::waitKey();
  #endif

  // Empirical evaluation
  segments.clear();
  groups.clear();

  SegmentClusters groundTruthClusters;
  SegmentClusteringXmlParser parser(UPM_GROUND_TRUTH_TELEFONICA_FILE);
  cv::Size domainSize;
  std::string imgName;
  parser.readXmlFile(segments, groundTruthClusters, domainSize, imgName);

  merger.setImageSize(domainSize);
  validator->setDomainSize(domainSize);
  merger.mergeSegments(segments, groupedSegments, groups);

  std::vector<std::tuple<unsigned int, unsigned int, double> > goodClustersFScore;
  double recallOfFScore = recallOfFScores(groundTruthClusters,
                                          groups,
                                          goodClustersFScore);
  ASSERT_GE(recallOfFScore, 0.48);

}

TEST(UnitGreedyMerger, ProjectPoint) {
  GreedyMerger merger(cv::Size(20, 20));
  // y = 10
  cv::Vec3d l(0, 1, -10);
  cv::Point2f proy = getProjectionPtn(l, cv::Point(4, 1));
  ASSERT_DOUBLE_EQ(4, proy.x);
  ASSERT_DOUBLE_EQ(10, proy.y);

  l = cv::Vec3d(0.707106781187, 0.707106781187, -5);
  proy = getProjectionPtn(l, cv::Point(2, 2));

  const double NEEDED_PRECISION = 0.000001;
  ASSERT_NEAR(3.535533905933, proy.x, NEEDED_PRECISION);
  ASSERT_NEAR(3.535533905933, proy.y, NEEDED_PRECISION);
}

//TEST(UnitGreedyMerger, BaseSegAdd) {
//  Segments ELSs = {
//      Segment(10, 10, 110, 10),
//      Segment(210, 10, 300, 10),
//      Segment(200, 10, 250, 10)
//  };
//  GreedyBaseSegClassicLS seg(&ELSs);
//  seg.addNewSegment(0);
//  Segment result = seg.getBaseSeg();
//  Segment expected(10, 10, 110, 10);
//  ASSERT_DOUBLE_EQ(expected[0], result[0]);
//  ASSERT_DOUBLE_EQ(expected[1], result[1]);
//  ASSERT_DOUBLE_EQ(expected[2], result[2]);
//  ASSERT_DOUBLE_EQ(expected[3], result[3]);
//
//  seg.addNewSegment(1);
//  result = seg.getBaseSeg();
//  expected = Segment(10, 10, 300, 10);
//  ASSERT_DOUBLE_EQ(expected[0], result[0]);
//  ASSERT_DOUBLE_EQ(expected[1], result[1]);
//  ASSERT_DOUBLE_EQ(expected[2], result[2]);
//  ASSERT_DOUBLE_EQ(expected[3], result[3]);
//
//  seg.addNewSegment(2);
//  result = seg.getBaseSeg();
//  expected = Segment(10, 10, 300, 10);
//  ASSERT_DOUBLE_EQ(expected[0], result[0]);
//  ASSERT_DOUBLE_EQ(expected[1], result[1]);
//  ASSERT_DOUBLE_EQ(expected[2], result[2]);
//  ASSERT_DOUBLE_EQ(expected[3], result[3]);
//
//}

//TEST(UnitGreedyMerger, GreedyBaseSegClassicLSFit) {
//  Segments corners = {
//      Segment(540.088, 60.6067, 638.467, 18.9333),
//      Segment(502.951, 77.713, 478.147, 88.176),
//      Segment(384.738, 126.603, 372.122, 131.29),
//      Segment(437.128, 102.189, 385.172, 125.883)
//  };
//
//  GreedyBaseSegClassicLS baseSeg(&corners);
//  baseSeg.addNewSegment(0);
//  baseSeg.addNewSegment(1);
//  baseSeg.addNewSegment(2);
//  baseSeg.addNewSegment(3);
//
//  cv::Vec3d expectedLine(0.387657874454789, 0.92180332629753237, -265.44771411308733);
//  Segment expectedBaseSeg(638.655823f, 19.3834991f, 372.188385f, 131.444489f);
//
//  cv::Vec3d line = baseSeg.getLineEquation();
//  Segment generatedBaseSeg = baseSeg.getBaseSeg();
//
//  ASSERT_EQ(expectedLine, line);
//  ASSERT_EQ(expectedBaseSeg, generatedBaseSeg);
//}
//
//TEST(UnitGreedyMerger, SingleSegmentsOverlap) {
//  Segments segments = {
//      // Base segment
//      Segment(100, 250, 100, 350),
//      // No overlapping
//      Segment(110, 220, 120, 240),
//      Segment(90, 390, 110, 450),
//      // Overlapping
//      Segment(140, 250, 140, 280),
//      Segment(110, 320, 120, 400),
//      Segment(110, 290, 120, 200),
//      Segment(100, 250, 100, 350),
//  };
//  GreedyBaseSegClassicLS base_seg(&segments);
//  base_seg.addNewSegment(0);
//
//  double NEEDED_PRECISION = 0.00000001;
//  ASSERT_NEAR(base_seg.segmentsOverlap(segments[0], segments[1]), 0, NEEDED_PRECISION);
//  ASSERT_NEAR(base_seg.segmentsOverlap(segments[0], segments[2]), 0, NEEDED_PRECISION);
//  ASSERT_NEAR(base_seg.segmentsOverlap(segments[0], segments[3]), 1, NEEDED_PRECISION);
//  ASSERT_NEAR(base_seg.segmentsOverlap(segments[0], segments[4]), 3.0 / 8.0, NEEDED_PRECISION);
//  ASSERT_NEAR(base_seg.segmentsOverlap(segments[0], segments[5]), 4.0 / 9.0, NEEDED_PRECISION);
//  ASSERT_NEAR(base_seg.segmentsOverlap(segments[0], segments[6]), 1, NEEDED_PRECISION);
//}
//
//TEST(UnitGreedyMerger, SegmentsOverlap) {
//  Segments segments = {
//      // Base segment
//      Segment(100, 100, 100, 200),
//      Segment(100, 250, 100, 350),
//      // No overlapping
//      Segment(110, 220, 120, 240),
//      Segment(90, 10, 110, 50),
//      // Overlapping
//      Segment(140, 150, 140, 180),
//      Segment(110, 320, 120, 400),
//      Segment(110, 290, 120, 200),
//      Segment(110, 10, 90, 180),
//  };
//  GreedyBaseSegClassicLS base_seg(&segments);
//  base_seg.addNewSegment(0);
//  base_seg.addNewSegment(1);
//
//  ASSERT_FALSE(base_seg.overlaps(2));
//  ASSERT_FALSE(base_seg.overlaps(3));
//  ASSERT_TRUE(base_seg.overlaps(4));
//  ASSERT_TRUE(base_seg.overlaps(5));
//  ASSERT_TRUE(base_seg.overlaps(6));
//
//  #ifdef _DEBUG_GTK
//
//  cv::Mat img(500, 300, CV_8UC3, cv::Scalar(255, 255, 255));
//  for (int i = 0; i < segments.size(); i++) {
//    if (i < 2) {
//      // Base segment
//      cv::segment(img, segments[i], CV_RGB(0, 0, 255), 2);
//    } else {
//      // Candidate segments
//      if (base_seg.overlaps(i)) {
//        cv::segment(img, segments[i], CV_RGB(255, 0, 0));
//      } else {
//        cv::segment(img, segments[i], CV_RGB(0, 255, 0));
//      }
//    }
//  }
//  cv::imshow("Segments", img);
//  cv::waitKey();
//  #endif
//}
//
//TEST(UnitGreedyMerger, SegmentsOverlapRandom) {
//  Segments segments = {
//      // Base segment
//      Segment(244, 320, 258, 277),
//      Segment(267, 249, 280, 213),
//      // No overlapping
//      Segment(232, 355, 215, 408),
//      Segment(206, 250, 232, 248),
//      // Overlapping
//      Segment(268, 285, 262, 303),
//      Segment(214, 282, 211, 303),
//      Segment(253, 410, 292, 277),
//  };
//  GreedyBaseSegClassicLS base_seg(&segments);
//  base_seg.addNewSegment(0);
//  base_seg.addNewSegment(1);
//
//  ASSERT_FALSE(base_seg.overlaps(2));
//  ASSERT_FALSE(base_seg.overlaps(3));
//  ASSERT_TRUE(base_seg.overlaps(4));
//  ASSERT_TRUE(base_seg.overlaps(5));
//  ASSERT_TRUE(base_seg.overlaps(6));
//
//  #ifdef _DEBUG_GTK
//
//  cv::Mat img(500, 500, CV_8UC3, cv::Scalar(255, 255, 255));
//  for (int i = 0; i < segments.size(); i++) {
//    if (i < 2) {
//      // Base segment
//      cv::segment(img, segments[i], CV_RGB(0, 0, 255), 2);
//    } else {
//      // Candidate segments
//      if (base_seg.overlaps(i)) {
//        cv::segment(img, segments[i], CV_RGB(255, 0, 0));
//      } else {
//
//        cv::segment(img, segments[i], CV_RGB(0, 255, 0));
//      }
//    }
//  }
//  cv::imshow("Segments", img);
//  cv::waitKey();
//  #endif
//}