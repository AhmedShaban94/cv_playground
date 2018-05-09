#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	CascadeClassifier face_classifier = CascadeClassifier("C:\\opencv3.1\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");
	VideoCapture stream(0);
	while (1)
	{
		//input video stream 
		Mat image;
		stream >> image;

		//convert image to grey-scale to be classified 
		Mat grey;
		cvtColor(image, grey, COLOR_BGR2GRAY);

		//face detection process 
		vector<Rect> faces;
		face_classifier.detectMultiScale(grey, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		if (faces.size() == 0)
		{
			std::cout << "No faces detected \n";
			cv::putText(image, "No faces detected", cv::Point(15, 15), 1, 1, cv::Scalar(0, 255, 0));
		}
		for (int i = 0; i < faces.size(); i++)
		{
			rectangle(image,
				Point(faces.at(i).x, faces.at(i).y),
				Point(faces.at(i).x + faces.at(i).width,
					faces.at(i).y + faces.at(i).height),
				Scalar(255, 0, 0), 2);
		}
		imshow("image", image);
		if (waitKey(30) == 13)
			break;
	}
	stream.release();
	destroyAllWindows();
	return EXIT_SUCCESS;
}