#include <iostream>
#include <random>
#include "GreedyMerger.h"
#include "LsdOpenCV.h"
#include "Utils.h"

using namespace upm;

namespace cv {
inline void
segments(cv::Mat img,
         const upm::Segments& segs,
         const cv::Scalar &color,
         int thickness = 1,
         int lineType = cv::LINE_8,
         int shift = 0) {
  for (const upm::Segment &seg: segs)
    cv::line(img, cv::Point2f(seg[0], seg[1]), cv::Point2f(seg[2], seg[3]), color, thickness, lineType, shift);
}
}

void
drawClusters(cv::Mat &img,
             const upm::Segments &segs,
             const upm::SegmentClusters &clusters,
             int thickness = 2,
             cv::Scalar color = {0, 0, 0},
             bool drawLineCluster = true,
             int lineType = cv::LINE_AA,
             int shift = 0) {
  bool random_c = color == cv::Scalar(0, 0, 0);
  std::mt19937 mt(0);
  std::uniform_real_distribution<double> dist(1.0, 255.0);
  for (const std::vector<unsigned int> &cluster: clusters) {
    if (random_c) {
      color = cv::Scalar( dist(mt), dist(mt), dist(mt));
    }

    if (cluster.size() > 1 && drawLineCluster) {
      cv::Vec3d l = upm::totalLeastSquareFitSegmentEndPts(segs, cluster);

      cv::Point2f max_p(0, 0), min_p(img.cols, img.rows);
      for (unsigned int idx: cluster) {
        cv::Point2f pp;
        pp = upm::getProjectionPtn(l, cv::Point2f(segs[idx][0], segs[idx][1]));
        if (pp.x > max_p.x) max_p = pp;
        if (pp.x < min_p.x) min_p = pp;

        pp = upm::getProjectionPtn(l, cv::Point2f(segs[idx][2], segs[idx][3]));
        if (pp.x > max_p.x) max_p = pp;
        if (pp.x < min_p.x) min_p = pp;
      }
      cv::line(img, min_p, max_p, color, round(thickness / 3.0), lineType, shift);
    }
    for (unsigned int idx: cluster) {
      cv::segments(img, {segs[idx]}, color, thickness, lineType, shift);
    }
  }
}

int main() {
  std::cout << "******************************************************" << std::endl;
  std::cout << "******************** FSG main demo *******************" << std::endl;
  std::cout << "******************************************************" << std::endl;

  // Read input image
  cv::Mat img = cv::imread("../images/P1080079.jpg");
  if (img.empty()) {
    std::cerr << "Error: Cannot read input image" << std::endl;
    return -1;
  }

  // Initialize the line segment merger
  GreedyMerger merger(img.size());

  //Detect lines
  upm::Segments detectedSegments;
  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  cv::Mat img2 = img.clone();

  // Detect segments with LSD
  LsdOpenCV().detect(gray, detectedSegments);

  std::cout << "Detected " << detectedSegments.size() << " line segments with LSD" << std::endl;
  cv::segments(img, detectedSegments, CV_RGB(255, 0, 0), 1);
  cv::imshow("Detected line segments", img);
  cv::imwrite("../Detected_line_segments.png", img);

  // Detect the segment clusters
  SegmentClusters detectedClusters;
  Segments mergedLines;
  merger.mergeSegments(detectedSegments, mergedLines, detectedClusters);
  drawClusters(img, detectedSegments, detectedClusters, 2);
  cv::imshow("Segment groups", img);
  cv::imwrite("../Segment_groups.png", img);

  // Get large lines from groups of segments
  Segments filteredSegments, noisySegs;
  filterSegments(detectedSegments, detectedClusters, filteredSegments, noisySegs);
  cv::segments(img2, filteredSegments, CV_RGB(0, 255, 0));
  cv::segments(img2, noisySegs, CV_RGB(255, 0, 0));
  cv::imshow("Obtained lines", img2);
  cv::imwrite("../Obtained_lines.png", img);

  cv::waitKey();
}