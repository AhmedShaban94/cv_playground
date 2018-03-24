#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace cv;

void warpTriangle(Mat &img1, Mat &img2, std::vector<Point2f> &t1, std::vector<Point2f> &t2);
static void calculateDelaunayTriangles(Rect rect, std::vector<Point2f> &points, std::vector< std::vector<int> > &delaunayTri);
void applyAffineTransform(Mat& warp_img, Mat& src, std::vector<Point2f>&  srcTri, std::vector<Point2f>& dstTri);
std::vector<Point2f> readPoints(std::string filename);

int main(int argc, char** argv)
{
	//Read input images 
	//std::string& image_name1 = "ted_cruz.jpg"; 
	//std::string& image_name2 = "donald_trump.jpg"; 

	Mat img1 = imread("ted_cruz.jpg");
	Mat img2 = imread("donald_trump.jpg");
	Mat img1_warped = img2.clone();

	//Read points 
	std::vector<Point2f> points1, points2;
	points1 = readPoints("ted_cruz.jpg.txt");
	points2 = readPoints("donald_trump.jpg.txt");

	//Convert Mat to float datatype 
	img1.convertTo(img1, CV_32F);
	img1_warped.convertTo(img1_warped, CV_32F);

	//Find convex hull 
	std::vector<Point2f> hull1;
	std::vector<Point2f> hull2;
	std::vector<int> hullIndex;

	convexHull(points2, hullIndex, false, false);

	for (int i = 0; i < hullIndex.size(); i++)
	{
		hull1.push_back(points1[hullIndex[i]]);
		hull2.push_back(points2[hullIndex[i]]);
	}

	//Find delauny trainglation for points on convex hull 
	std::vector<std::vector<int> > dt;
	Rect rect(0, 0, img1_warped.cols, img1_warped.rows);
	calculateDelaunayTriangles(rect, hull2, dt);

	//Apply affine transform to delanuay triangles 
	for (size_t i = 0; i < dt.size(); i++)
	{
		std::vector<Point2f> t1, t2;
		//Get points for img1, img2 corresponding to the trianglution 
		for (size_t j = 0; j < 3; j++)
		{
			t1.push_back(hull1[dt[i][j]]);
			t2.push_back(hull2[dt[i][j]]);
		}

		warpTriangle(img1, img1_warped, t1, t2);
	}

	//Calculate mask 
	std::vector<Point> hull8U;
	for (int i = 0; i < hull2.size(); i++)
	{
		Point pt(hull2[i].x, hull2[i].y);
		hull8U.push_back(pt);
	}

	Mat mask = Mat::zeros(img2.rows, img2.cols, img2.depth());
	fillConvexPoly(mask, &hull8U[0], hull8U.size(), Scalar(255, 255, 255));

	// Clone seamlessly 
	Rect r = boundingRect(hull2);
	Point center = (r.tl() + r.br()) / 2;

	Mat output;
	img1_warped.convertTo(img1_warped, CV_8UC3);
	seamlessClone(img1_warped, img2, mask, center, output, NORMAL_CLONE);

	imshow("Face Swapped", output);
	if (waitKey(0) == 13)
	{
		destroyAllWindows();
	}

	return EXIT_SUCCESS;
}


//Read points from text files 
std::vector<Point2f> readPoints(std::string filename)
{
	std::vector<Point2f> points;
	std::ifstream file(filename.c_str());
	float x, y;
	int count = 0;
	while (file >> x >> y)
	{
		points.push_back(Point2f(x, y));
	}

	return points;
}

//Apply affine transform 
void applyAffineTransform(Mat& warp_img, Mat& src, std::vector<Point2f>&  srcTri, std::vector<Point2f>& dstTri)
{
	// Given a pair of trinagles, find the affine transform. 
	Mat warpMat = getAffineTransform(srcTri, dstTri);

	// Apply the affine transform 
	warpAffine(src, warp_img, warpMat, warp_img.size(), INTER_LINEAR, BORDER_REFLECT);
}

// Calculate Delaunay triangles for set of points
// Returns the vector of indices of 3 points for each triangle
static void calculateDelaunayTriangles(Rect rect, std::vector<Point2f> &points, std::vector< std::vector<int> > &delaunayTri)
{

	// Create an instance of Subdiv2D
	Subdiv2D subdiv(rect);

	// Insert points into subdiv
	for (std::vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
		subdiv.insert(*it);

	std::vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	std::vector<Point2f> pt(3);
	std::vector<int> ind(3);

	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		pt[0] = Point2f(t[0], t[1]);
		pt[1] = Point2f(t[2], t[3]);
		pt[2] = Point2f(t[4], t[5]);

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
			for (int j = 0; j < 3; j++)
				for (size_t k = 0; k < points.size(); k++)
					if (abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
						ind[j] = k;

			delaunayTri.push_back(ind);
		}
	}

}


// Warps and alpha blends triangular regions from img1 and img2 to img
void warpTriangle(Mat &img1, Mat &img2, std::vector<Point2f> &t1, std::vector<Point2f> &t2)
{

	Rect r1 = boundingRect(t1);
	Rect r2 = boundingRect(t2);

	// Offset points by left top corner of the respective rectangles
	std::vector<Point2f> t1Rect, t2Rect;
	std::vector<Point> t2RectInt;
	for (int i = 0; i < 3; i++)
	{

		t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
		t2RectInt.push_back(Point(t2[i].x - r2.x, t2[i].y - r2.y)); // for fillConvexPoly

	}

	// Get mask by filling triangle
	Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
	fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

	// Apply warpImage to small rectangular patches
	Mat img1Rect;
	img1(r1).copyTo(img1Rect);

	Mat img2Rect = Mat::zeros(r2.height, r2.width, img1Rect.type());

	applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);

	multiply(img2Rect, mask, img2Rect);
	multiply(img2(r2), Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
	img2(r2) = img2(r2) + img2Rect;
}
