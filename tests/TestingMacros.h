/**
 * @copyright 2018 Xoan Iago Suarez Canosa. All rights reserved.
 * Constact: iago.suarez.canosa@alumnos.upm.es
 * Software developed in the PhD: Augmented Reality for Urban Environments
 */
#ifndef LINE_EXPERIMENTS_TESTINGMACROS_H_
#define LINE_EXPERIMENTS_TESTINGMACROS_H_

#include <gtest/gtest.h>
#include <iostream>

#define UPM_NOT_IMPLEMENTED_EXCEPTION \
    std::cerr << "Error: Function Not Implemented" << std::endl; \
    throw std::logic_error("Error: Function Not Implemented");

#define UPM_SEGS_TOLERANCE 0.1
#define UPM_ROT_RECT_TOLERANCE 0.001

static inline bool SegmentEquality(upm::Segment s1, upm::Segment s2, double tolerance) {
  return (std::abs(s1[0] - s2[0]) < tolerance || std::abs(s1[0] - s2[2]) < tolerance) &&
      (std::abs(s1[1] - s2[1]) < tolerance || std::abs(s1[1] - s2[3]) < tolerance) &&
      (std::abs(s1[2] - s2[2]) < tolerance || std::abs(s1[2] - s2[0]) < tolerance) &&
      (std::abs(s1[3] - s2[3]) < tolerance || std::abs(s1[3] - s2[1]) < tolerance);
}

#define ASSERT_USORTED_EDGE_EQ(e_a, e_b) ASSERT_PRED2(upm::compareSets<upm::Pixel>, e_a, e_b);

#define ASSERT_SEGMENT_NEAR(s_a, s_b, tolerance) ASSERT_PRED3(SegmentEquality, s_a, s_b, tolerance);

#define ASSERT_SEGMENTS_NEAR(segs_a, segs_b, tol) \
    ASSERT_EQ((segs_a).size(), (segs_b).size()); \
    for (int i = 0; i < (segs_a).size(); i++) { \
      ASSERT_SEGMENT_NEAR((segs_a)[i], (segs_b)[i], tol); \
    }

#define ASSERT_SEGMENT_EQ(s_a, s_b) ASSERT_SEGMENT_NEAR((s_a), (s_b), UPM_SEGS_TOLERANCE)

#define ASSERT_VECTOR_NEAR(v1, v2, precision) \
{ \
  ASSERT_EQ(v1.size(), v2.size()); \
  for(int i = 0; i < v1.size() ; i++){ \
    ASSERT_NEAR(v1[i], v2[i], precision); \
  } \
}

#define ASSERT_SEGMENTS_EQ(_a, _b) \
    ASSERT_EQ((_a).size(), (_b).size()); \
    for (int i = 0; i < (_a).size(); i++) { \
      const Segment& s_a = (_a)[i]; \
      const Segment& s_b = (_b)[i]; \
      ASSERT_SEGMENT_EQ(s_a, s_b) \
    }

#define ASSERT_CLUSTERS_EQ(a, b) \
    ASSERT_EQ((a).size(), (b).size()); \
    for (int i = 0; i < (a).size(); i++) { \
      ASSERT_EQ((a)[i].size(), (b)[i].size()); \
      for (int j = 0; j < (a)[i].size(); j++) { \
        ASSERT_EQ((a)[i][j], (b)[i][j]); \
      } \
    }

#define ASSERT_EDGE_EQ(edge1, edge2) \
   ASSERT_EQ(edge1.size(), edge2.size()); \
   ASSERT_FALSE(edge1.empty()); \
   if((edge1[0].x == edge2[0].x) && (edge1[0].y == edge2[0].y)){ \
     for(int i = 0; i < edge1.size() ; i++){ \
       ASSERT_EQ(edge1[i].x, edge2[i].x); \
       ASSERT_EQ(edge1[i].y, edge2[i].y); \
     } \
   } else { \
     for(int i = 0; i < edge1.size() ; i++){ \
       ASSERT_EQ(edge1[i].x, edge2[edge1.size() - 1 - i].x); \
       ASSERT_EQ(edge1[i].y, edge2[edge1.size() - 1 - i].y); \
     } \
   }

#define ASSERT_EDGES_EQ(edges1, edges2) \
   ASSERT_EQ(edges1.size(), edges2.size()); \
  for (int i = 0; i < edges1.size(); i++) { \
    ASSERT_EQ(edges1[i].size(), edges2[i].size()); \
    auto *edge1 = &edges1[i]; \
    auto *edge2 = &edges2[i]; \
    int currentPos = i; \
    while (edge1->front() != edge2->front() && edge1->size() == edge2->size()) { \
      currentPos++; \
      edge2 = &edges2[currentPos]; \
    } \
    if (edge1->front() != edge2->front()) { \
      currentPos = i; \
      edge2 = &edges2[currentPos]; \
      while (edge1->front() != edge2->front() && edge1->size() == edge2->size()) { \
        currentPos--; \
        edge2 = &edges2[currentPos]; \
      } \
    } \
    for (int j = 0; j < edges1[i].size(); j++) { \
      ASSERT_EQ(edge1->at(j).x, edge2->at(j).x); \
      ASSERT_EQ(edge1->at(j).y, edge2->at(j).y); \
    } \
  }

#define ASSERT_EQ_ROTRECT(a, b)\
    ASSERT_NEAR((a).size.width, (b).size.width, UPM_ROT_RECT_TOLERANCE); \
    ASSERT_NEAR((a).size.height, (b).size.height, UPM_ROT_RECT_TOLERANCE); \
    ASSERT_NEAR((a).center.x, (b).center.x, UPM_ROT_RECT_TOLERANCE); \
    ASSERT_NEAR((a).center.y, (b).center.y, UPM_ROT_RECT_TOLERANCE); \
    ASSERT_NEAR((a).angle, (b).angle, UPM_ROT_RECT_TOLERANCE);

#define UPM_CVMAT_FLOAT_PRECISION 0.0001

#define ASSERT_CVMAT_EQ(a, b) \
    ASSERT_EQ((a).type(), (b).type());\
    ASSERT_EQ((a).channels(), (b).channels());\
    ASSERT_EQ((a).size(), (b).size());\
    for (int col = 0; col < (a).cols; col++) {\
      for (int row = 0; row < (a).rows; row++) {\
        switch ((a).type()) {\
          case CV_64FC1:\
            ASSERT_NEAR((a).at<double>(row, col), (b).at<double>(row, col), UPM_CVMAT_FLOAT_PRECISION);\
            break;\
          case CV_32FC1:\
            ASSERT_NEAR((a).at<float>(row, col), (b).at<float>(row, col), UPM_CVMAT_FLOAT_PRECISION);\
            break;\
          case CV_32SC1:\
            ASSERT_EQ((a).at<int>(row, col), (b).at<int>(row, col));\
            break;\
          case CV_16SC1:\
            ASSERT_EQ((a).at<short>(row, col), (b).at<short>(row, col));\
            break;\
          case CV_8UC1:\
            ASSERT_EQ((a).at<uchar>(row, col), (b).at<uchar>(row, col));\
            break;\
          default:\
          UPM_NOT_IMPLEMENTED_EXCEPTION;\
        }\
      }\
    }

#define ASSERT_CVMAT_NEAR(a, b, precision) \
    ASSERT_EQ((a).type(), (b).type());\
    ASSERT_EQ((a).channels(), (b).channels());\
    ASSERT_EQ((a).size(), (b).size());\
    for (int col = 0; col < (a).cols; col++) {\
      for (int row = 0; row < (a).rows; row++) {\
        switch ((a).type()) {\
          case CV_64FC1:\
            ASSERT_NEAR((a).at<double>(row, col), (b).at<double>(row, col), precision);\
            break;\
          case CV_32FC1:\
            ASSERT_NEAR((a).at<float>(row, col), (b).at<float>(row, col), precision);\
            break;\
          case CV_32SC1:\
            ASSERT_EQ((a).at<int>(row, col), (b).at<int>(row, col));\
            break;\
          case CV_16SC1:\
            ASSERT_EQ((a).at<short>(row, col), (b).at<short>(row, col));\
            break;\
          case CV_8UC1:\
            ASSERT_EQ((a).at<uchar>(row, col), (b).at<uchar>(row, col));\
            break;\
          default:\
          UPM_NOT_IMPLEMENTED_EXCEPTION;\
        }\
      }\
    }


#define ASSERT_VP_NEAR(vp1, vp2, tolerance) ASSERT_NEAR(1, std::abs((vp1).dot(vp2)), tolerance)

static inline bool cvPointsNear(cv::Point2f p1, cv::Point2f p2, double tolerance) {
  return cv::norm(p1 - p2) < tolerance;
}

#define ASSERT_CV_POINT_NEAR(p1, p2, tolerance) ASSERT_PRED3(cvPointsNear, p1, p2, tolerance)

#define ASSERT_CV_POINTS_NEAR(pts1, pts2, tolerance) \
  ASSERT_EQ((pts1).size(), (pts2).size()); \
  for(int i = 0 ; i < (pts1).size() ; i++) ASSERT_CV_POINT_NEAR((pts1)[i], (pts2)[i], tolerance)

#define ASSERT_CVMAT_OF_DBL_NEAR(mat1, mat2, tolerance) \
  ASSERT_EQ(mat1.size(), mat2.size()); \
  ASSERT_EQ(mat1.type(), CV_64FC1); \
  ASSERT_EQ(mat1.type(), mat2.type()); \
  for (int row = 0; row < mat1.rows; row++) { \
    for (int col = 0; col < mat1.cols; col++) { \
      if (std::isinf(mat1.at<double>(row, col))) { \
        ASSERT_TRUE(std::isinf(mat2.at<double>(row, col))); \
      } else { \
        ASSERT_NEAR(mat1.at<double>(row, col), mat2.at<double>(row, col), tolerance); \
      } \
    } \
  }

#endif  // LINE_EXPERIMENTS_TESTINGMACROS_H_
