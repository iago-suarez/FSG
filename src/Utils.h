#ifndef FSG_MATH_H
#define FSG_MATH_H

#include <iostream>
#include <opencv2/opencv.hpp>

namespace upm {
// Define some data types
typedef cv::Vec4f Segment;
typedef std::vector<Segment> Segments;
typedef std::vector<unsigned int> SegmentsCluster;
typedef std::vector<SegmentsCluster> SegmentClusters;

/**
 * @brief Calculates the length of a line segment
 * @param s The input segment
 * @return The length of the segment
 */
inline float
segLength(const Segment &s) {
  // The optimal way to do that, is to compute first the differences
  const float dx = s[0] - s[2];
  const float dy = s[1] - s[3];
  // And after that, do the square root, avoiding the use of double
  return std::sqrt(dx * dx + dy * dy);
}

/**
 * @brief Calculates the orientation of a line segment
 * @param s The input segment
 * @return The orientation of the segment in radians: [-pi/2 ; pi/2]
 */
inline float
segOrientation(const Segment &s) {
  cv::Vec3d l = cv::Vec3d(s[0], s[1], -1).cross(cv::Vec3d(s[2], s[3], -1));
  l /= std::sqrt(l[0] * l[0] + l[1] * l[1]);
  // Since the polar equation is x*cos(theta) + y*sin(theta) = rho
  // And theta is defined in the range [-pi/2 ; pi/2), cos(theta) will never
  // be a negative number, and so, we should invert the equation symbol if it's negative
  if (l[0] < 0) {
    // If the distance is negative change the symbol of the equation
    l *= -1;
  }
  // We use std::asin[1] instead std::acos[0] because it returns angles in the range [-pi/2 ; pi/2]
  double theta = std::atan2(l[1], l[0]);
  if (theta >= M_PI_2) theta -= M_PI;
  if (theta < -M_PI_2) theta += M_PI;
  return theta;
}

/**
 * @brief Computes the projection of the point p into the line l.
 * @param l Line equation as (a, b, c) where ax + by + c = 0
 * @param p 2D point that will be projected into the line
 * @return The 2D projected point
 */
static cv::Point2f getProjectionPtn(const cv::Vec3f &l, const cv::Point2f &p) {
  const cv::Vec3f homoP(p.x, p.y, 1);
  if (l.dot(homoP) == 0) {
    // If the point is over the line return this same point
    return p;
  }
  // Since the direction of l is (-l.b, l.a), the rotated 90 degrees vector will be: (l.a, l.b)
  // The direction vector of the perpendicular rect we want to calc.
  const cv::Vec2f v2(l[0], l[1]);
  // Rect with direction v2 passing by point p
  const cv::Vec3f r2(v2[1], -v2[0], v2[0] * p.y - v2[1] * p.x);
  // return the intersection between the two lines in cartesian coordinates
  const cv::Vec3f p2 = l.cross(r2);
  return {p2[0] / p2[2], p2[1] / p2[2]};
}

/**
 * @brief Fits a line to the points using the least square method.
 * @note This method cannot adjust vertical lines!!
 * @param points The points to which we want to adjust the line.
 * @return The general equation of the line. [a, b, c] where: ax + by + c = 0
 */
static cv::Vec3d
totalLeastSquareFit(const std::vector<cv::Point2f> &points) {
  if (points.size() < 2) {
    std::cerr << "Error: a minimum of 2 points are required" << std::endl;
    throw std::invalid_argument("Error: a minimum of 2 points are required");
  }
  double N = points.size();
  double x_mean = 0, y_mean = 0;
  double num = 0, den = 0;

  for (const cv::Point2f &p: points) {
    x_mean += p.x;
    y_mean += p.y;
  }
  x_mean /= N;
  y_mean /= N;

  double dx, dy;
  for (const cv::Point2f &p: points) {
    dx = p.x - x_mean;
    dy = p.y - y_mean;
    num += -2.0 * dx * dy;
    den += (dy * dy) - (dx * dx);
  }

  double theta = 0.5 * std::atan2(num, den);
  double rho = x_mean * std::cos(theta) + y_mean * std::sin(theta);
  cv::Vec3d l(std::cos(theta), std::sin(theta), -rho);
  return l;
}

/**
 * @brief Fits a line to the endpoints of the segments using the total least square method.
 * @param segments
 * @param selectedSegments
 * @return
 */
static cv::Vec3d
totalLeastSquareFitSegmentEndPts(const Segments &segments,
                                 const std::vector<unsigned int> &selectedSegments = {}) {
  if (segments.size() < 1) {
    std::cerr << "Error: a minimum of 2 points are required" << std::endl;
    throw std::invalid_argument("Error: a minimum of 2 points are required");
  }

  std::vector<cv::Point2f> pts;
  if (selectedSegments.empty()) {
    // Add all the segments
    for (const Segment &s: segments) {
      pts.emplace_back(s[0], s[1]);
      pts.emplace_back(s[2], s[3]);
    }
  } else {
    for (unsigned int i: selectedSegments) {
      const Segment &s = segments[i];
      pts.emplace_back(s[0], s[1]);
      pts.emplace_back(s[2], s[3]);
    }
  }
  return totalLeastSquareFit(pts);
}

}

#endif //FSG_MATH_H
