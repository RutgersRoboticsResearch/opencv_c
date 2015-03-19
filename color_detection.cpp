#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;


int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;


const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

const int MAX_NUM_OBJECTS=50;

const int MIN_OBJECT_AREA = 20*20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;


const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";

void on_trackbar( int, void* )
{//This function gets called whenever a
  // trackbar position is changed
}

string intToString(int number){


  std::stringstream ss;
  ss << number;
  return ss.str();
}

void createTrackbars(){
  //create window for trackbars

  namedWindow(trackbarWindowName,0);
  //create memory to store trackbar name on window
  char TrackbarName[50];
  cout << TrackbarName << "H_MIN" << H_MIN << endl;
  cout << TrackbarName << "H_MAX" << H_MAX << endl;
  cout << TrackbarName << "S_MIN" << S_MIN << endl;
  cout << TrackbarName << "S_MAX" << S_MAX << endl;
  cout << TrackbarName << "V_MIN" << V_MIN << endl;
  cout << TrackbarName << "V_MAX" << V_MAX << endl;
  
  //create trackbars and insert them into window
  //3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
  //the max value the trackbar can move (eg. H_HIGH), 
  //and the function that is called whenever the trackbar is moved(eg. on_trackbar)
  //                                  ---->    ---->     ---->      

  createTrackbar( "H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar );
  createTrackbar( "H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar );
  createTrackbar( "S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar );
  createTrackbar( "S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar );
  createTrackbar( "V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar );
  createTrackbar( "V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar );
}

void drawObject(int x, int y,Mat &frame){

  //use some of the openCV drawing functions to draw crosshairs
  //on your tracked image!

    //UPDATE:JUNE 18TH, 2013
    //added 'if' and 'else' statements to prevent
    //memory errors from writing off the screen (ie. (-25,-25) is not within the window!)
  circle(frame,Point(x,y),20,Scalar(0,255,0),2);
  if(y-25>0)
    line(frame,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
  else line(frame,Point(x,y),Point(x,0),Scalar(0,255,0),2);
  if(y+25<FRAME_HEIGHT)
    line(frame,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
  else line(frame,Point(x,y),Point(x,FRAME_HEIGHT),Scalar(0,255,0),2);
  if(x-25>0)
    line(frame,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
  else line(frame,Point(x,y),Point(0,y),Scalar(0,255,0),2);
  if(x+25<FRAME_WIDTH)
    line(frame,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);
  else line(frame,Point(x,y),Point(FRAME_WIDTH,y),Scalar(0,255,0),2);

  putText(frame,intToString(x)+","+intToString(y),Point(x,y+30),1,1,Scalar(0,255,0),2);

}


void morphOps(Mat &thresh){

  //create structuring element that will be used to "dilate" and "erode" image.
  //the element chosen here is a 3px by 3px rectangle

  Mat erodeElement = getStructuringElement( MORPH_RECT,Size(3,3));

  //dilate with larger element so make sure object is nicely visible
  Mat dilateElement = getStructuringElement( MORPH_RECT,Size(8,8));

  erode(thresh,thresh,erodeElement);
  erode(thresh,thresh,erodeElement);

  dilate(thresh,thresh,dilateElement);
  dilate(thresh,thresh,dilateElement);
}


//hide the local functions in an anon namespace
namespace {
  void help(char** av) {
    cout << "The program captures frames from a video file, image sequence (01.jpg, 02.jpg ... 10.jpg) or camera connected to your computer." << endl
    << "Usage:\n" << av[0] << " <video file, image sequence or device number>" << endl
    << "q,Q,esc -- quit" << endl
    << "space   -- save frame" << endl << endl
    << "\tTo capture from a camera pass the device number. To find the device number, try ls /dev/video*" << endl
    << "\texample: " << av[0] << " 0" << endl
    << "\tYou may also pass a video file instead of a device number" << endl
    << "\texample: " << av[0] << " video.avi" << endl
    << "\tYou can also pass the path to an image sequence and OpenCV will treat the sequence just like a video." << endl
    << "\texample: " << av[0] << " right%%02d.jpg" << endl;
  }
}

  void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed) {

    Mat temp;
    threshold.copyTo(temp);

//these two vectors needed for output of findContours
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;

//find contours of filtered image using openCV findContours function
    findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );

//use moments method to find our filtered object
    double refArea = 0;
    bool objectFound = false;
    if (hierarchy.size() > 0) {

      int numObjects = hierarchy.size();

      //if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
      if(numObjects<MAX_NUM_OBJECTS){
        for (int index = 0; index >= 0; index = hierarchy[index][0]) {

          Moments moment = moments((cv::Mat)contours[index]);
          double area = moment.m00;

      //if the area is less than 20 px by 20px then it is probably just noise
      //if the area is the same as the 3/2 of the image size, probably just a bad filter
      //we only want the object with the largest area so we safe a reference area each
      //iteration and compare it to the area in the next iteration.
          if(area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea){
            x = moment.m10/area;
            y = moment.m01/area;
            objectFound = true;
            refArea = area;
          }else objectFound = false;
        }

      //let user know you found an object
        if(objectFound ==true){

          putText(cameraFeed,"Tracking Object",Point(0,50),2,1,Scalar(0,255,0),2);

        //draw object location on screen
          drawObject(x,y,cameraFeed);}
        }else{ 

          putText(cameraFeed,"TOO MUCH NOISE! ADJUST FILTER",Point(0,50),1,2,Scalar(0,0,255),2);
        }
      }
    }

    int main(int ac, char** av) {

  //some boolean variables for different functionality within this
  //program
      bool trackObjects = true;
      bool useMorphOps = true;

  //Matrix to store each frame of the webcam feed
      Mat cameraFeed;

  //matrix storage for HSV image
      Mat HSV;

  //matrix storage for binary threshold image
      Mat threshold;

  //x and y values for the location of the object
      int x=0, y=0;

  //create slider bars for HSV filtering
      createTrackbars();

      //video capture object to acquire webcam feed
      VideoCapture capture;

      //open capture object at location zero (default location for webcam)
      capture.open(0);

      //set height and width of capture frame
      capture.set(500,FRAME_WIDTH);
      capture.set(500,FRAME_HEIGHT);

      //start an infinite loop where webcam feed is copied to cameraFeed matrix
      //all of our operations will be performed within this loop
      while(1){

        //store image to matrix
        capture.read(cameraFeed);

        //convert frame from BGR to HSV colorspace
        cvtColor(cameraFeed,HSV,COLOR_BGR2HSV);

        //filter HSV image between values and store filtered image to
        //threshold matrix
        inRange(HSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),threshold);

        //perform morphological operations on thresholded image to eliminate noise
        //and emphasize the filtered object(s)
        if(useMorphOps)
          morphOps(threshold);

        //pass in thresholded frame to our object tracking function
        //this function will return the x and y coordinates of the
        //filtered object
        if(trackObjects)
          trackFilteredObject(x,y,threshold,cameraFeed);

    //show frames 
        imshow(windowName2,threshold);
        imshow(windowName,cameraFeed);
        imshow(windowName1,HSV);

    //delay 30ms so that screen can refresh.
    //image will not appear without this waitKey() command
        char key = (char)waitKey(30);

        switch(key){
          case 'q':
          return 0;
          default:
          break;
        }
      }
      return 0;
    }
