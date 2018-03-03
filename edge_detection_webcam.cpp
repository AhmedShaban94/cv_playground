#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int, char**)
{
	VideoCapture stream(0);


	if (!stream.isOpened())
		cout << "No camera was detected \n";
	namedWindow("video recording", 1);
	Mat edges;
	for (;;)
	{
		Mat image;
		cv::Mat flipped_image;
		stream >> image;
		flip(image, flipped_image, 1);
		/*imshow("video recording", flipped_image); */
		cvtColor(flipped_image, edges, COLOR_BGR2GRAY);
		GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		Canny(edges, edges, 0, 30, 3);
		imshow("video recording", edges);
		//waitKey function must have 25 or 30 input to capture image for 25 or 30 millisec
		//otherwise it will just capture the first image only. 
		if (waitKey(30) >= 0) break;
	}
	stream.release();
	destroyAllWindows();
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}