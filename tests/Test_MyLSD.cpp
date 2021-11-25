/**
 * @copyright 2018 Xoan Iago Suarez Canosa. All rights reserved.
 * Constact: iago.suarez.canosa@alumnos.upm.es
 * Software developed in the PhD: Augmented Reality for Urban Environments
 */

#include "LsdOpenCV.h"
#include "SegmentClusteringXmlParser.h"
#include <gtest/gtest.h>

using namespace upm;

#define MY_FLOAT_PRECISSION 0.01 // pixels

Segments
readSegmentsFile(const std::string &filename) {
  Segments segments;

  //load lines
  std::fstream lineFile;
  lineFile.open(filename, std::ios::in);
  if (!lineFile.is_open()) {
    std::stringstream ss;
    ss << "Can't open the input line file: " << filename;
    std::cerr << ss.str() << std::endl;
    throw std::invalid_argument(ss.str());
  }

  int templineID;
  Segment tempVec1;

  // Read line file
  while (!lineFile.eof()) {
    lineFile >> templineID;
    if (lineFile.eof()) {
      break;
    }
    lineFile >> tempVec1[0] >> tempVec1[1] >> tempVec1[2] >> tempVec1[3];
    segments.push_back(tempVec1);
  }

  return segments;
}

/**
 * @brief This test compares the output of the ported LSD algorithm (LsdOpenCV)
 * with the OpenCV original output stored for the MadridBuildings database.
 * @param argc
 * @param argv
 * @return
 */
TEST(MyLSD, MainTest) {
  LsdOpenCV detector(cv::LSD_REFINE_NONE);

  std::string imageFilename = "/home/iago/workspace/line-experiments/resources/datasets/MadridBuildings/telefonica/original.jpg";
  cv::Mat image = cv::imread(imageFilename, cv::IMREAD_GRAYSCALE);

  Segments expectedLines, detectedLines;
//  SegmentClusteringXmlParser parser("/home/iago/workspace/line-experiments/resources/datasets/MadridBuildings/telefonica/clusters.xml");
  std::string txtSegments = "/home/iago/workspace/line-experiments/resources/datasets/MadridBuildings/telefonica/output.txt";

  std::ifstream lineDirPath;
  lineDirPath.open(txtSegments);

  std::string line;
  expectedLines = readSegmentsFile(txtSegments);

  SegmentClusters expectedClusters;
  std::string img_name;

  if (image.empty()) {
    std::cerr << "Error reading image: \"" << imageFilename << "\"" << std::endl;
    throw std::invalid_argument("Error reading image");
  }

  //Detect lines
  detectedLines = detector.detect(image);

  const float NEEDED_PRECISSION = 0.1;

  ASSERT_EQ(expectedLines.size(), detectedLines.size());
  for (int i = 0; i < expectedLines.size(); ++i) {
    std::cout << "Comparing: " << expectedLines[i] << " and " << detectedLines[i] << std::endl;

    ASSERT_NEAR(expectedLines[i][0], detectedLines[i][0], NEEDED_PRECISSION);
    ASSERT_NEAR(expectedLines[i][1], detectedLines[i][1], NEEDED_PRECISSION);
    ASSERT_NEAR(expectedLines[i][2], detectedLines[i][2], NEEDED_PRECISSION);
    ASSERT_NEAR(expectedLines[i][3], detectedLines[i][3], NEEDED_PRECISSION);
  }

}