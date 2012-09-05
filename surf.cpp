#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/legacy/compat.hpp>

#include <iostream>
#include <vector>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int, char**)
{
  // Open the default video camera
  VideoCapture cap(0);
  if(!cap.isOpened())
  {
    cerr << "Video stream could not be located." << endl;
    return -1;
  }

  // Open the source image that will be used for detection
  Mat objDef = imread("IMG_0337.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  if(!objDef.data)
  {
    cerr << "Error loading the object definition image." << endl;
    return -1;
  }

  // Create the SURF detector and extractor
  SurfFeatureDetector detector(500);
  SurfDescriptorExtractor extractor;

  // Vectors for the object definition keypoints
  // as well as the keypoints for each frame in the video
  vector<KeyPoint> keysObjDef, keysFrame;

  // cv::Mat to hold SURF definitions
  Mat descObjDef, descFrame;

  // Run SURF on the object definition
  detector.detect(objDef, keysObjDef);
  extractor.compute(objDef, keysObjDef, descObjDef);

  // This object will match descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  vector<DMatch> matches, goodMatches;
  vector<DMatch>::iterator goodMatch;

  // These are points that define a good mapping around the detected object
  vector<Point2f> objPnts, framePnts;
  vector<Point2f> objCorners(4), frameCorners(4);

  // Get the corners from the source image
  objCorners[0] = cvPoint(0,0);
  objCorners[1] = cvPoint(objDef.cols, 0);
  objCorners[2] = cvPoint(objDef.cols, objDef.rows);
  objCorners[3] = cvPoint( 0, objDef.rows );

  // Storage locations for each captured video frame
  // one for the full color, the other for the converted grayscale
  Mat frame, gray, H;

  // Some storage
  double maxDist = 0, minDist = 100;

  for(;;)
  {
    // Reset for the next loop
    objPnts.clear();
    framePnts.clear();

    cap >> frame; // get a new frame from camera
    cvtColor(frame, gray, CV_BGR2GRAY);

    // Extract SURF points by initializing parameters
    detector.detect(frame, keysFrame);
    extractor.compute(frame, keysFrame, descFrame);

    // Matching descriptor vectors using FLANN matcher
    matcher.match(descObjDef, descFrame, matches);

    // Quick calculation of max and min distances between keypoints
    for(int i = 0; i < descObjDef.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < minDist ) minDist = dist;
      if( dist > maxDist ) maxDist = dist;
    }

    // Determine "good" matches (i.e., matches whose distances is less than 3*minDist)
    for( int i = 0; i < descObjDef.rows; i++ )
    { if( matches[i].distance < 3*minDist )
       { goodMatches.push_back( matches[i]); }
    }

    for(goodMatch = goodMatches.begin(); goodMatch != goodMatches.end(); ++goodMatch)
    {
      objPnts.push_back(keysObjDef[goodMatch->queryIdx].pt);
      framePnts.push_back(keysFrame[goodMatch->trainIdx].pt);
    }

    H = findHomography(objPnts, framePnts, CV_RANSAC);

    perspectiveTransform(objCorners, frameCorners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line(frame, frameCorners[0] + Point2f(objDef.cols, 0), frameCorners[1] + Point2f(objDef.cols, 0), Scalar(0, 255, 0), 4);
    line(frame, frameCorners[1] + Point2f(objDef.cols, 0), frameCorners[2] + Point2f(objDef.cols, 0), Scalar(0, 255, 0), 4);
    line(frame, frameCorners[2] + Point2f(objDef.cols, 0), frameCorners[3] + Point2f(objDef.cols, 0), Scalar(0, 255, 0), 4);
    line(frame, frameCorners[3] + Point2f(objDef.cols, 0), frameCorners[0] + Point2f(objDef.cols, 0), Scalar(0, 255, 0), 4);

    //-- Show detected matches
    imshow("Good Matches & Object detection", frame);

    if(waitKey(30) >= 0) break;
  }
  // the camera will be deinitialized automatically in VideoCapture destructor

  cvDestroyWindow("edges");
  return 0;
}
