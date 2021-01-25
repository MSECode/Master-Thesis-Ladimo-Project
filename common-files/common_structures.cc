#include "common_structures.h"


void LookupTable::setLookupMatrix(const int& rows_number, const int& points_on_width, const int& points_on_height)
{
	std::vector<int> temp_vector(rows_number);
	std::iota(std::begin(temp_vector), std::end(temp_vector), 0);

	lookup_matrix = cv::Mat(points_on_height, points_on_width, CV_32S, temp_vector.data()).t();
}

cv::Mat LookupTable::getLookupMatrix()
{
	return lookup_matrix.t();
}

bool isPixelInsideImage(cv::Point2i& left_pixel_position, cv::Point2i& right_pixel_position, cv::Mat& left_matrix_limits, cv::Mat& right_matrix_limits)
{
	bool is_inside = false;
	if (left_pixel_position.y > left_matrix_limits.at<int>(0, 0) && left_pixel_position.y < left_matrix_limits.at<int>(0, 1) &&
		left_pixel_position.x > left_matrix_limits.at<int>(1, 0) && left_pixel_position.x < left_matrix_limits.at<int>(1, 1) &&
		right_pixel_position.y > right_matrix_limits.at<int>(0, 0) && right_pixel_position.y < right_matrix_limits.at<int>(0, 1) &&
		right_pixel_position.x > right_matrix_limits.at<int>(1, 0) && right_pixel_position.x < right_matrix_limits.at<int>(1, 1)) is_inside = true;
	return is_inside;
}

cv::Mat defineMatrixLimits(int top, int bottom, int left, int right)
{
	std::array<std::array<int, 2>, 2> limits_array = { {{top, bottom}, {left, right}} };
	return cv::Mat(2, 2, CV_32S, limits_array.data()).t();
}
