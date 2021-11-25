/**
 * @copyright 2018 Xoan Iago Suarez Canosa. All rights reserved.
 * Constact: iago.suarez.canosa@alumnos.upm.es
 * Software developed in the PhD: Augmented Reality for Urban Environments
 */
#include "SegmentClusteringXmlParser.h"

namespace upm {
SegmentClusteringXmlParser::SegmentClusteringXmlParser(const std::string &filename) :
    mOutputFilename(filename) {
}

void SegmentClusteringXmlParser::createOutputFile(cv::Mat &img,
                                                  Segments &segments,
                                                  const cv::Size imgSize,
                                                  const std::string &imageName) {
  SalientSegments salientSegs(segments.size());
  for (int i = 0; i < segments.size(); i++) {
    salientSegs[i].segment = segments[i];
    salientSegs[i].salience = 0;
  }
  createOutputFile(img, salientSegs, imgSize, imageName, false);
}


void SegmentClusteringXmlParser::createOutputFile(cv::Mat &img,
                                                  SalientSegments &segments,
                                                  const cv::Size imgSize,
                                                  const std::string &imageName,
                                                  bool saliency) {
  assert(!segments.empty() || (!img.empty() && img.size() == imgSize));

  std::cout << "Creating segments clustering results file: \"" << mOutputFilename << "\"" << std::endl;

  // One XMLDocument is created when the struct SegClusterData is instantiated,
  // so lest fill this document

  //Create the file header with the current time
  time_t tm;
  time(&tm);
  struct tm *t2 = localtime(&tm);
  char buf[1024];
  strftime(buf, sizeof(buf) - 1, " Segment Clustering file. Generated %c ", t2);

  mXmlDoc.InsertFirstChild(mXmlDoc.NewDeclaration());
  mXmlDoc.InsertEndChild(
      mXmlDoc.NewComment("*************************************************************"));
  mXmlDoc.InsertEndChild(mXmlDoc.NewComment(buf));
  mXmlDoc.InsertEndChild(
      mXmlDoc.NewComment("*************************************************************"));

  // Add the root element and add it to the document
  mpRoopt = mXmlDoc.NewElement("segment_clustering");
  mXmlDoc.InsertEndChild(mpRoopt);

  tinyxml2::XMLElement *pElement;

  pElement = mXmlDoc.NewElement("image_name");
  pElement->SetText(imageName.c_str());
  mpRoopt->InsertFirstChild(pElement);

  pElement = mXmlDoc.NewElement("image_size");
  std::stringstream ss;
  ss << imgSize.width << ", " << imgSize.height;
  pElement->SetText(ss.str().c_str());
  mpRoopt->InsertFirstChild(pElement);

  // The segments element will be a child of segment_clustering that contains
  // a list of the segments detected by the LSD method in the image img
  pElement = mXmlDoc.NewElement("segments");

  // Detect the segments and save then to the XML file
//  if (segments.empty()) {
//    cv::Mat gray;
//    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
//    cv::Ptr<cv::LineSegmentDetector> lsd_detector = createMyLineSegmentDetector(cv::LSD_REFINE_ADV);
//    lsd_detector->detect(gray, segments);
//  }

  // For each detected segment, save it to the XML
  for (int i = 0; i < segments.size(); i++) {
    // Create the XML element for the i-th segment
    tinyxml2::XMLElement *pListElement = mXmlDoc.NewElement("segment");
    // The id of the segment is only its index in the segments array
    tinyxml2::XMLElement *pSegmentId = mXmlDoc.NewElement("id");
    pSegmentId->SetText(i);
    pListElement->InsertEndChild(pSegmentId);

    // The endpoints will be a list of points like: [1.23, 3.3, 100.1, 234.11]
    tinyxml2::XMLElement *pSegmentEndpoints = mXmlDoc.NewElement("endpoints");
    std::stringstream ss;
    // The << operator is overloaded to print in the correct format, so use it

    constexpr int NDECIMALS = 6;
    ss << "[" << std::fixed << std::setprecision(NDECIMALS) << segments[i].segment[0] << ", "
       << std::fixed << std::setprecision(NDECIMALS) << segments[i].segment[1] << ", "
       << std::fixed << std::setprecision(NDECIMALS) << segments[i].segment[2] << ", "
       << std::fixed << std::setprecision(NDECIMALS) << segments[i].segment[3] << "]";
    pSegmentEndpoints->SetText(ss.str().c_str());
    pListElement->InsertEndChild(pSegmentEndpoints);

    //Inset the segment XML element in the segments XML element
    pElement->InsertEndChild(pListElement);

    if (saliency) {
      // The endpoints will be a list of points like: [1.23, 3.3, 100.1, 234.11]
      tinyxml2::XMLElement *pSegmentSaliency = mXmlDoc.NewElement("saliency");
      pSegmentSaliency->SetText(segments[i].salience);
      pListElement->InsertEndChild(pSegmentSaliency);
    }

    //Inset the segment XML element in the segments XML element
    pElement->InsertEndChild(pListElement);
  }
  // Add an attribute with the number of segments
  pElement->SetAttribute("numberOfSegments", int(segments.size()));
  // Insert the generated list in the root element segment_clustering
  mpRoopt->InsertEndChild(pElement);

  // Create an empty section that is going to contain the clusters generated by the application
  mpClustersXlmElem = mXmlDoc.NewElement("clusters");
  mpRoopt->InsertEndChild(mpClustersXlmElem);

  // Write the file and check errors
  tinyxml2::XMLError eResult = mXmlDoc.SaveFile(mOutputFilename.c_str());
  XMLCheckResult(eResult);
}

inline std::vector<std::string>
split(const std::string &s, char delim) {
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> elems;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
    // elems.push_back(std::move(item)); // if C++11 (based on comment from @mchiasson)
  }
  return elems;
}

inline std::vector<unsigned int>
parseUnsignedIntList(std::string input) {
  input.erase(std::remove(input.begin(), input.end(), '['), input.end());
  input.erase(std::remove(input.begin(), input.end(), ']'), input.end());
  input.erase(std::remove(input.begin(), input.end(), ' '), input.end());

  std::vector<unsigned int> result;
  for (auto chunk : split(input, ',')) {
    result.push_back(atoi(chunk.c_str()));
  }
  return result;
}

std::vector<double>
parseDoubleList(std::string input) {
  input.erase(std::remove(input.begin(), input.end(), '['), input.end());
  input.erase(std::remove(input.begin(), input.end(), ']'), input.end());
  input.erase(std::remove(input.begin(), input.end(), ' '), input.end());

  std::vector<double> result;
  for (auto chunk : split(input, ',')) {
    result.push_back(atof(chunk.c_str()));
  }
  return result;
}

void SegmentClusteringXmlParser::readXmlFile(Segments &segments,
                                             SegmentClusters &clusters,
                                             cv::Size &img_size,
                                             std::string &img_name) {
  std::cout << "Reading existing results file" << std::endl;
  tinyxml2::XMLError eResult = mXmlDoc.LoadFile(mOutputFilename.c_str());
  XMLCheckResult(eResult);

  // Get the root XML element <segment_clustering> using LastChild
  mpRoopt = mXmlDoc.LastChild();
  if (mpRoopt == nullptr) {
    std::cerr << "Error reading the root node of the XML document \"" << mOutputFilename << "\"" << std::endl;
    throw std::runtime_error("Error reading the root node of the XML document");
  }

  tinyxml2::XMLElement *pElement;
  pElement = mpRoopt->FirstChildElement("image_name");
  if (pElement == nullptr) {
    std::cerr << "Error reading the \"image_name\" node of the XML document \"" << mOutputFilename << "\"" << std::endl;
    throw std::runtime_error("Error reading the \"image_name\" node of the XML document");
  }
  img_name = pElement->GetText();

  pElement = mpRoopt->FirstChildElement("image_size");
  if (pElement == nullptr) {
    std::cerr << "Error reading the \"img_size\" node of the XML document \"" << mOutputFilename << "\"" << std::endl;
    throw std::runtime_error("Error reading the \"img_size\" node of the XML document");
  }
  auto size_vec = parseUnsignedIntList(pElement->GetText());
  assert(size_vec.size() == 2);
  img_size = cv::Size(size_vec[0], size_vec[1]);

  // Parse the list of segments
  pElement = mpRoopt->FirstChildElement("segments");
  if (pElement == nullptr) {
    std::cerr << "Error reading the segments node of the XML document \"" << mOutputFilename << "\"" << std::endl;
    throw std::runtime_error("Error reading the segments node of the XML document");
  }

  // Parse each of the <segment> elements
  tinyxml2::XMLElement *pListElement = pElement->FirstChildElement("segment");
  while (pListElement != nullptr) {
    tinyxml2::XMLElement *pEndpointsElement = pListElement->FirstChildElement("endpoints");
    if (pEndpointsElement == nullptr) {
      std::cerr << "Error reading the endpoints node of the XML document \"" << mOutputFilename << "\"" << std::endl;
      throw std::runtime_error("Error reading the endpoints node of the XML document");
    }
    // readText should be a list of endpoints in format [1.23, 3.3, 100.1, 234.11]
    const char *readText = pEndpointsElement->GetText();
    if (readText == nullptr) {
      std::cerr << "Error reading the endpoints node content of the XML document \"" << mOutputFilename << "\"" << std::endl;
      throw std::runtime_error("Error reading the enpoints node content of the XML document");
    }
    // Covert from string to list of doubles
    auto vecVal = parseDoubleList(readText);
    assert(vecVal.size() == 4);
    // Generate the segment with the read values and add it to the list of segments
    Segment endpoints(vecVal[0], vecVal[1], vecVal[2], vecVal[3]);
    segments.push_back(endpoints);
    // Advance to the next element
    pListElement = pListElement->NextSiblingElement("segment");
  }

  // Parse the list of clusters
  pElement = mpRoopt->FirstChildElement("clusters");
  if (pElement == nullptr) {
    std::cerr << "Error reading the clusters node of the XML document \"" << mOutputFilename << "\"" << std::endl;
    throw std::runtime_error("Error reading the segments node of the XML document");
  }

  // Parse each of the <cluster> elements
  pListElement = pElement->FirstChildElement("cluster");
  while (pListElement != nullptr) {
    // readText should be a list of positive integers like: [23, 45, 1, 453, ... ]
    const char *readText = pListElement->GetText();
    if (readText == nullptr) {
      std::cerr << "Error reading the cluster node content of the XML document \"" << mOutputFilename << "\"" << std::endl;
      throw std::runtime_error("Error reading the enpoints node content of the XML document");
    }
    // Convert the string to numbers and add it to the vector in clusters
    std::vector<unsigned int> vecVal = parseUnsignedIntList(readText);
    clusters.push_back(vecVal);
    // Advance to the next element
    pListElement = pListElement->NextSiblingElement("cluster");
  }
}

void SegmentClusteringXmlParser::saveClustersToXmlFile(const SegmentClusters &clusters) {

//  std::cout << "Saving data to file \"" << mOutputFilename << "\"" << std::endl;
//
//  // Delete all the file clusters
//  tinyxml2::XMLElement *pElement = mpRoopt->FirstChildElement("clusters");
//  pElement->DeleteChildren();
//
//  // Add the clusters to XML file
//  for (const auto &cluster : clusters) {
//    tinyxml2::XMLElement *pListElement = mXmlDoc.NewElement("cluster");
//
//    std::stringstream ss;
//    ss << cluster;
//    pListElement->SetText(ss.str().c_str());
//    pElement->InsertEndChild(pListElement);
//  }
//
//  pElement->SetAttribute("numberOfClusters", int(clusters.size()));
//
//  // Save file
//  tinyxml2::XMLError eResult = mXmlDoc.SaveFile(mOutputFilename.c_str());
//  XMLCheckResult(eResult);
}
}
