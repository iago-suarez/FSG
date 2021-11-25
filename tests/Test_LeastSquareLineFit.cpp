/**
 * @copyright 2018 Xoan Iago Suarez Canosa. All rights reserved.
 * Constact: iago.suarez.canosa@alumnos.upm.es
 * Software developed in the PhD: Augmented Reality for Urban Environments
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "Utils.h"

using namespace upm;

inline cv::Mat
toHomoMat(const cv::Mat &M) {
  cv::Mat result(M.rows + 1, M.cols, M.type());
  for (int col = 0; col < M.cols; col++) {
    for (int row = 0; row < M.rows; row++) {
      if (M.type() == CV_32FC1)
        result.at<float>(row, col) = M.at<float>(row, col);
      else if (M.type() == CV_64FC1)
        result.at<double>(row, col) = M.at<double>(row, col);
      else return cv::Mat();
    }
    if (M.type() == CV_32FC1)
      result.at<float>(M.rows, col) = 1.0f;
    else if (M.type() == CV_64FC1)
      result.at<double>(M.rows, col) = 1.0;
  }
  return result;
}


double
calculateLSTotError(std::vector<cv::Point2f> &corners, cv::Vec3f l) {
  cv::Mat pts(corners.size(), 2, CV_32FC1, corners.data());
  pts = toHomoMat(pts.t());
  // Nx1 = Nx3 * 3x1
  cv::Mat tmp = cv::Mat(l).t() * pts;
  tmp = tmp * tmp.t();
  return tmp.at<float>(0, 0);
}

TEST(LeastSquareLineFit, ClassicLeastSquare_RealLine1) {
  std::vector<cv::Point2f> corners = {
      cv::Point2f(118.96861128f, 296.40912454f),
      cv::Point2f(1.84417227f, 169.36650754f),
      cv::Point2f(141.31031418f, 329.10691009f),
      cv::Point2f(18.799824f, 200.25815612f)
  };

  cv::Vec3f line;
  double expectedError;

}

TEST(LeastSquareLineFit, ClassicLeastSquare_RealLine2) {
  std::vector<cv::Point2f> corners = {
      cv::Point2f(540.088, 60.6067), cv::Point2f(638.467, 18.9333),
      cv::Point2f(502.951, 77.713), cv::Point2f(478.147, 88.176),
      cv::Point2f(384.738, 126.603), cv::Point2f(372.122, 131.29),
      cv::Point2f(437.128, 102.189), cv::Point2f(385.172, 125.883)
  };

  cv::Vec3d expectedLine(0.38769373307577526, 0.92178824538706805, -265.4630942571942);
  cv::Vec3d line = totalLeastSquareFit(corners);

  ASSERT_EQ(expectedLine, line);
//  #ifdef _DEBUG_GTK
//  cv::Mat img(480, 640, CV_8UC3, CV_RGB(255, 255, 255));
//  for (cv::Point2f p: corners)
//    cv::circle(img, p, 2, CV_RGB(255, 0, 0), -1);
//  drawLineEq(img, line, CV_RGB(0, 255, 0), 1);
//  cv::imshow("Classic Fit", img, 800);
//  cv::waitKey();
//  #endif

}

TEST(LeastSquareLineFit, FitVerticalLine) {
  std::vector<cv::Point2f> corners = {
      cv::Point2f(200, 50),
      cv::Point2f(200, 200),
      cv::Point2f(200, 250),
      cv::Point2f(200, 400)
  };

  cv::Vec3d expectedLine(1, 0, -200);
  double expectedError = 0;

  cv::Vec3d line = totalLeastSquareFit(corners);
  double error = 0;

  ASSERT_NEAR(expectedError, error, 0.001);
  ASSERT_NEAR(expectedLine[0], line[0], 0.001);
  ASSERT_NEAR(expectedLine[1], line[1], 0.001);
  ASSERT_NEAR(expectedLine[2], line[2], 0.001);

  corners = {
      cv::Point2f(199, 50),
      cv::Point2f(201, 200),
      cv::Point2f(201, 250),
      cv::Point2f(199, 400)
  };

  expectedError = 4;

  line = totalLeastSquareFit(corners);
  error = calculateLSTotError(corners, line);

  ASSERT_NEAR(expectedError, error, 0.001);
  ASSERT_NEAR(expectedLine[0], line[0], 0.001);
  ASSERT_NEAR(expectedLine[1], line[1], 0.001);
  ASSERT_NEAR(expectedLine[2], line[2], 0.001);

}