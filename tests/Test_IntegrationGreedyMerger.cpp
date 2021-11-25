/**
 * @copyright 2018 Xoan Iago Suarez Canosa. All rights reserved.
 * Constact: iago.suarez.canosa@alumnos.upm.es
 * Software developed in the PhD: Augmented Reality for Urban Environments
 */
#include <gtest/gtest.h>
#include "SegmentClusteringXmlParser.h"
#include "GreedyMerger.h"
#include "TestingMacros.h"

using namespace upm;

TEST(IntegrationGreedyMerger, FSGDetectSavedClustersTelefonica) {

  SegmentClusteringXmlParser parser("/home/iago/workspace/line-experiments/resources/tests/IntegrationGreedyMerger_FSGTelefonica.xml");
  Segments xmlLsdSegs;
  SegmentClusters xmlClusters, detectedClusters;

  // Read the XML file
  cv::Size imgSize;
  std::string imgName;
  parser.readXmlFile(xmlLsdSegs, xmlClusters, imgSize, imgName);

  // Initialize the merger
  GreedyMerger merger(imgSize);
  merger.setEndPointError(6);
  merger.setIncludeSmallSegments(true);
  merger.setEndpointsMaxDistance(4);
  merger.setMinSegSize(0.01);
  merger.setNumLengthBins(1000);
  merger.setNumOrientationBins(50);

  // Detect the segment clusters
  Segments nonUsedSegs;
  merger.mergeSegments(xmlLsdSegs, nonUsedSegs, detectedClusters);

//  cv::Mat img;
//  SegmentClusteringXmlParser parser2(upm::getResource("/tests/IntegrationGreedyMerger_FSGTelefonica.xml"));
//  parser2.createOutputFile(img, xmlLsdSegs, imgSize, "original.jpg");
//  parser2.saveClustersToXmlFile(detectedClusters);

  ASSERT_CLUSTERS_EQ(xmlClusters, detectedClusters);
}

TEST(IntegrationGreedyMerger, FSGDetectSavedClustersChessboard) {

  SegmentClusteringXmlParser parser("/home/iago/workspace/line-experiments/resources/tests/IntegrationGreedyMerger_FSGChessboard.xml");
  Segments xmlLsdSegs;
  SegmentClusters xmlClusters, detectedClusters;

  // Read the XML file
  cv::Size imgSize;
  std::string imgName;
  parser.readXmlFile(xmlLsdSegs, xmlClusters, imgSize, imgName);

  // Initialize the merger
  GreedyMerger merger(imgSize);
  merger.setEndPointError(6);
  merger.setIncludeSmallSegments(true);
  merger.setEndpointsMaxDistance(4);
  merger.setMinSegSize(0.01);
  merger.setNumLengthBins(1000);
  merger.setNumOrientationBins(50);

  // Detect the segment clusters
  Segments nonUsedSegs;
  merger.mergeSegments(xmlLsdSegs, nonUsedSegs, detectedClusters);

//  cv::Mat img;
//  SegmentClusteringXmlParser parser2(upm::getResource("/tests/IntegrationGreedyMerger_FSGChessboard.xml"));
//  parser2.createOutputFile(img, xmlLsdSegs, imgSize, "chessboard.jpg");
//  parser2.saveClustersToXmlFile(detectedClusters);

  ASSERT_CLUSTERS_EQ(xmlClusters, detectedClusters);
}