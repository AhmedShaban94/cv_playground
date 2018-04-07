#include <opencv2\opencv.hpp>
#include <opencv2\face.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <Windows.h>

using namespace cv;
using namespace cv::face;

struct return_face_extractor
{
	std::vector<Rect> faces;
	Mat cropped_face;
};

std::vector<std::string> get_files(std::string folder)
{
	std::vector<std::string> names;
	std::string search_path = folder + "/*.*";
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}


return_face_extractor face_extractor(Mat image, CascadeClassifier classifier)
{
	return_face_extractor pack;
	Mat grey;
	cvtColor(image, grey, COLOR_BGR2GRAY, 0);
	std::vector<Rect> faces;
	classifier.detectMultiScale(grey, faces, 1.3, 5);
	//crop all found faces 
	Mat cropped_image;
	for (auto face : faces)
	{
		cropped_image = image(face);
	}
	pack.cropped_face = cropped_image;
	pack.faces = faces;
	return pack;
}

CascadeClassifier create_training_data()
{
	CascadeClassifier classifiers = CascadeClassifier("C:\\opencv_3.4\\etc\\haarcascades\\haarcascade_frontalface_alt.xml");
	VideoCapture stream = VideoCapture(0);
	int count = 0;
	while (true)
	{
		count++;
		Mat frame;
		stream >> frame;
		Mat face;
		//Mat face_grey; 
		return_face_extractor pack = face_extractor(frame, classifiers);
		resize(pack.cropped_face, face, Size(200, 200));
		cvtColor(face, face, COLOR_BGR2GRAY);

		//saving training data 
		std::string file_name = "D:\\Master Computer Vision With OpenCV in Python\\08 Machine Learning in Computer Vision\\train\\" + std::to_string(count) + ".jpg";
		imwrite(file_name, face);

		//put count on images and display training faces 
		putText(face, std::to_string(count), Point(50, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0), 3);
		imshow("training image", face);

		if (waitKey(30) == 13 || count == 100)
			break;
	}
	destroyAllWindows();
	stream.release();
	return classifiers;
}

Ptr<face::LBPHFaceRecognizer> training_model(const std::string& training_file_path)
{
	std::string training_path = training_file_path;
	std::vector<std::string>files = get_files(training_path);
	std::vector<int> labels;
	std::vector<Mat> training_data;
	int i = 0;
	for (auto file : files)
	{
		i++;
		labels.push_back(i);
		Mat image = imread(training_path + file, 0);
		training_data.push_back(image);
	}

	std::cout << "Training started\n";
	Ptr<face::LBPHFaceRecognizer> model = face::LBPHFaceRecognizer::create();
	model->train(training_data, labels);
	std::cout << "Training finished\n";
	return model;
}

int main(int argc, char** argv)
{

	//step 1: Creating training data 
	CascadeClassifier classifier = create_training_data();

	//step 2: training model 
	Ptr <face::LBPHFaceRecognizer> model = training_model("D:\\Master Computer Vision With OpenCV in Python\\08 Machine Learning in Computer Vision\\train\\");
	VideoCapture stream = VideoCapture(0);
	while (true)
	{
		Mat frame;
		stream >> frame;
		return_face_extractor pack = face_extractor(frame, classifier);
		Mat face;
		cvtColor(pack.cropped_face, face, COLOR_BGR2GRAY);
		int label;
		double confidence;
		model->predict(face, label, confidence);
		std::string display_string = std::to_string(confidence) + "% sure it's the user\n";
		putText(frame, display_string, Point(50, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 0), 2);
		rectangle(frame, pack.faces.at(0), Scalar(255, 20, 0), 1, 8, 0);
		imshow("predicition", frame);
		if (waitKey(30) == 13)
			break;
	}
	stream.release();
	destroyAllWindows();
	return EXIT_SUCCESS;
}
