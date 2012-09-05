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
  VideoCapture cap(0); // open the default camera
  if(!cap.isOpened())  // check if we succeeded
    return -1;

  Mat frame, gray;
  CvMat colorMat, grayMat;
  namedWindow("edges",1);

  CvMemStorage* storage = cvCreateMemStorage(0);

  // Define sequence for storing surf keypoints and descriptors
  CvSeq *imageKeypoints = 0, *imageDescriptors = 0;

  // Define the SURF params
  CvSURFParams params = cvSURFParams(500, 1);

  int i = 0;
  static CvScalar red_color[] = {0, 0, 255};

  for(;;)
  {
    cap >> frame; // get a new frame from camera
    cvtColor(frame, gray, CV_BGR2GRAY);
    imshow("edges", gray);

    // Convert cv::Mat to CvMat
    colorMat = frame;
    grayMat  = gray;

    // Extract SURF points by initializing parameters
    cvExtractSURF(&grayMat, 0, &imageKeypoints, &imageDescriptors, storage, params);

    // draw the keypoints on the captured frame
    for(i = 0; i < imageKeypoints->total; i++)
    {
      CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem( imageKeypoints, i );
      CvPoint center;
      int radius;
      center.x = cvRound(r->pt.x);
      center.y = cvRound(r->pt.y);
      radius = cvRound(r->size*1.2/9.*2);
      cvCircle(&colorMat, center, radius, red_color[0], 1, 8, 0 );
    }
    cvShowImage("edges", &colorMat);

    if(waitKey(30) >= 0) break;
  }
  // the camera will be deinitialized automatically in VideoCapture destructor

  cvDestroyWindow("edges");
  return 0;
}
