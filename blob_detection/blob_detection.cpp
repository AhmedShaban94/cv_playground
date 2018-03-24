#include <opencv2/core/core.hpp>
#include <opencv\cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\features2d.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	SimpleBlobDetector::Params params;
	// Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 200;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 1500;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.1;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.87;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.5;
	
	cv::Mat image = cv::imread("red_eyes.jpg"); 
	cv::SimpleBlobDetector detector(params);
	std::vector<cv::KeyPoint> keypoints; 
	detector.detect(image, keypoints);
	cv::Mat blank; 
	drawKeypoints(image, keypoints, blank, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS); 
	cv::imshow("image with keypoints", blank); 
	cv::waitKey(); 
	return EXIT_SUCCESS; 
}