/**
 * @copyright 2018 Xoan Iago Suarez Canosa. All rights reserved.
 * Constact: iago.suarez.canosa@alumnos.upm.es
 * Software developed in the PhD: Augmented Reality for Urban Environments
 */
#include <gtest/gtest.h>
#include "FastAcontrarioValidator.h"
#include "GreedyMerger.h"
#include "Utils.h"
#include "SegmentClusteringXmlParser.h"

#define TELEFONICA_IMAGE "/home/iago/workspace/line-experiments/resources/datasets/MadridBuildings/telefonica/original.jpg"
#define UPM_DEGREES_TO_RADS 0.01745329252  // (M_PI / 180.0)

using namespace upm;

inline bool
isInsideRoratedRect(const cv::RotatedRect &R, const cv::Point2f &p) {
  const cv::Point2f v = p - R.center;
  const double angle = UPM_DEGREES_TO_RADS * (R.angle - 90);
  const double cos_angle = std::cos(angle);
  const double sin_angle = std::sin(angle);
  const cv::Point2f rotated_v(+cos_angle * v.x + sin_angle * v.y,
                              -sin_angle * v.x + cos_angle * v.y);
  return std::abs(rotated_v.y) <= R.size.width / 2.0 && std::abs(rotated_v.x) <= R.size.height / 2.0;
}


TEST(FastAcontrarioValidator, IntegralOfTheBoxSpace) {

  double len = 900;
  double rho = 20;
  double radio = 10;

  cv::RotatedRect R(cv::Point2f(200, 100), cv::Size2f(rho * 2, len), 60);

  double angle = UPM_DEGREES_TO_RADS * (R.angle - 90);
  double cos_angle = std::cos(angle);
  double sin_angle = std::sin(angle);
  cv::Vec3d l(sin_angle, -cos_angle, -sin_angle * R.center.x + cos_angle * R.center.y);

  long n_inside = 0;
  long n_tests = 0;

  cv::Point2f pts[4];
  R.points(pts);
  cv::Point2f v1 = pts[1] - pts[0];
  cv::Point2f v2 = pts[3] - pts[0];

  //For all the points in R
  for (double delta_x = 0; delta_x <= 1; delta_x += 0.005) {
    for (double delta_y = 0; delta_y <= 1; delta_y += 0.005) {

      cv::Point2f p = pts[0] + v1 * delta_x + v2 * delta_y;

      // We check how many of the pixels that are around this point are in R --> in_n_pts
      // and how many are not R --> out_n_pts
//      cv::Size2d domainSize(320, 240);
//      cv::Mat img(domainSize, CV_8UC3, CV_RGB(255, 255, 255));
//      drawRotatedRectangle(img, R, CV_RGB(255, 0, 255));
//      drawLineEq(img, l, CV_RGB(0, 0, 255), 1);
//      cv::circle(img, p, 1, CV_RGB(0, 0, 0), -1);

      for (double check_angle = 0; check_angle <= 2 * M_PI; check_angle += 0.05) {
        cv::Point2d point_to_check(p.x + std::cos(check_angle) * radio, p.y + std::sin(check_angle) * radio);

        bool inside = isInsideRoratedRect(R, point_to_check);
        if (inside) n_inside++;
        n_tests++;

//        cv::Vec3b color = inside ? cv::Vec3b(0, 255, 0) : cv::Vec3b(0, 0, 255);
//        img.at<cv::Vec3b>(point_to_check.y, point_to_check.x) = color;
      }
    }
  }

  double prob = n_inside / (double) n_tests;
  std::cout << "Real probability: " << prob << std::endl;

  double d = rho / 2.0;
  double theta1 = (rho + d < radio) ? std::asin((rho + d) / radio) : M_PI_2;
  double theta2 = (rho - d < radio) ? std::asin((rho - d) / radio) : M_PI_2;
  double theta = theta1 + theta2;
  std::cout << "Estimated probability: " << theta / (M_PI) << std::endl;

}
