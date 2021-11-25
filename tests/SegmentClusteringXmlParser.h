/**
 * @copyright 2018 Xoan Iago Suarez Canosa. All rights reserved.
 * Constact: iago.suarez.canosa@alumnos.upm.es
 * Software developed in the PhD: Augmented Reality for Urban Environments
 */
#ifndef LINE_EXPERIMENTS_SEGMENTCLUSTERINGXMLPARSER_H_
#define LINE_EXPERIMENTS_SEGMENTCLUSTERINGXMLPARSER_H_

#include <string>
#include <opencv2/opencv.hpp>
#include <Utils.h>
#include "tinyxml2.h"

#define UPM_GROUND_TRUTH_TELEFONICA_FILE "/home/iago/workspace/line-experiments/resources/datasets/MadridBuildings/telefonica/clusters.xml"

namespace upm {

struct SalientSegment {
  Segment segment;
  double salience;

  SalientSegment() = default;
  SalientSegment(const Segment &segment, double salience) : segment(segment), salience(salience) {}

  inline bool operator<(const SalientSegment &rhs) const {
    if (salience == rhs.salience) {
      float dx1 = segment[0] - segment[2];
      float dx2 = rhs.segment[0] - rhs.segment[2];
      float dy1 = segment[1] - segment[3];
      float dy2 = rhs.segment[1] - rhs.segment[3];
      return std::sqrt(dx1 * dx1 + dy1 * dy1) > std::sqrt(dx2 * dx2 + dy2 * dy2);
    } else {
      return salience > rhs.salience;
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const SalientSegment &segment) {
    os << "segment: " << segment.segment << " salience: " << segment.salience;
    return os;
  }
};

typedef std::vector<SalientSegment> SalientSegments;

// TODO Add the image size to the xml, and the method used to extract the lines
class SegmentClusteringXmlParser {
  std::string mOutputFilename;
  tinyxml2::XMLDocument mXmlDoc;
  tinyxml2::XMLNode *mpRoopt = nullptr;
  tinyxml2::XMLElement *mpClustersXlmElem = nullptr;

 public:

  SegmentClusteringXmlParser() = default;

  explicit SegmentClusteringXmlParser(const std::string &filename);

  void setOutputFilename(const std::string &filename) {
    mOutputFilename = filename;
  }

  /**
   * @brief Creates a XML file containing the segments detected in the image data.img
   * and an empty section for the segment clusters
   * @param data The GenerateSegmentClustersGT application data
   */
  /**
   * @brief Creates a XML file containing the segments detected in the image data.img
   * @param img If no segments are provided, they will be obtained calling LSD
   * in this image
   * @param segments The input segments, if it's a empty vector, it will be
   * filled with LDS-ADV segments
   * @param imgSize The size of the image where the segmentes were detected.
   * If segments is empty should be img.size()
   * @param imageName
   */
  void createOutputFile(cv::Mat &img,
                        Segments &segments,
                        const cv::Size imgSize,
                        const std::string &imageName);

  void createOutputFile(cv::Mat &img,
                        SalientSegments &segments,
                        const cv::Size imgSize,
                        const std::string &imageName,
                        bool saliency=false);
  /**
   * @brief Read a segments list and a clusters list from an XML file.
   * @param data The data structure where the read data is going to be written
   */
  void
  readXmlFile(Segments &segments,
              SegmentClusters &clusters,
              cv::Size &img_size,
              std::string &img_name);

  /**
   * @brief Save the data clusters from memory to the XML file specified in
   * output_filename.
   * @param data The Application Data
   */
  void saveClustersToXmlFile(const SegmentClusters &clusters);
};
}  // namespace upm

#endif  // LINE_EXPERIMENTS_SEGMENTCLUSTERINGXMLPARSER_H_
