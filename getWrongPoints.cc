#include "getWrongPoints.h"

cv::Mat GetWrongPoints::getWrongPoints(const cv::Mat& inputCloud)
{
	rows = inputCloud.rows;
	cv::Mat badDepths = cv::Mat(rows, 1, CV_32S);

	for (size_t n = 0; n < inputCloud.rows; n++)
	{
		if (inputCloud.at<float>(n,2) > 6000.f)
		{
			// bad points
			badDepths.at<int>(n, 0) = 1;
		}
		else
		{
			// good points
			badDepths.at<int>(n, 0) = 0;
		}
	}

	return badDepths;
}

