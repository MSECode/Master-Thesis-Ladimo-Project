#ifndef GETWRONGPOINTS
#define GETWRONGPOINTS

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

class GetWrongPoints {
	public:
		int rows;
		cv::Mat getWrongPoints(const cv::Mat& inputCloud);
};

#endif // !GETWRONGPOINTS
