#include "support_structures_simulated_grid.h"
#include "io_functions.h"


/**
 * define left and right calibration matrices
 * from the snippets of text previously red
 * data are transformed from string to double
 * and pushed into the matrices
 */
void CalibrationAndImagesDataset::setLeftRightCalibrationMatrices(const std::string& cam_0_, const std::string& cam_1_, const char& delete_char)
{
	std::vector<double> temp_vector_left;
	std::string temp_sub_string = cam_0_.substr(1, cam_0_.length() - 2);
	temp_sub_string.erase(std::remove(temp_sub_string.begin(), temp_sub_string.end(), delete_char), temp_sub_string.end());
	std::istringstream input_string_left(temp_sub_string);
	temp_vector_left.assign(std::istream_iterator<double>(input_string_left), std::istream_iterator<double>());
	cam_0 = cv::Mat(3, 3, CV_64F, temp_vector_left.data()).t();

	std::vector<double> temp_vector_right;
	temp_sub_string = cam_1_.substr(1, cam_1_.length() - 2);
	temp_sub_string.erase(std::remove(temp_sub_string.begin(), temp_sub_string.end(), delete_char), temp_sub_string.end());
	std::istringstream input_string_right(temp_sub_string);
	temp_vector_right.assign(std::istream_iterator<double>(input_string_right), std::istream_iterator<double>());
	cam_1 = cv::Mat(3, 3, CV_64F, temp_vector_right.data()).t();
}

void CalibrationAndImagesDataset::setDoffs(const std::string& doffs_)
{
	disparity_offset = std::stod(doffs_);
}

void CalibrationAndImagesDataset::setBaseline(const std::string& baseline_)
{
	camera_baseline = std::stod(baseline_);
}

void CalibrationAndImagesDataset::setWidth(const std::string& width_)
{
	image_width = std::stoi(width_);
}

void CalibrationAndImagesDataset::setHeight(const std::string& height_)
{
	image_height = std::stoi(height_);
}

void CalibrationAndImagesDataset::setLeftRightGroundTruth(const std::string& ground_truth_0, const std::string& ground_truth_1)
{
	std::vector<double> temp_left = io::readBinaryToDouble(ground_truth_0);
	std::vector<double> temp_right = io::readBinaryToDouble(ground_truth_1);

	gt_left = cv::Mat(image_width, image_height, CV_64F, temp_left.data()).t();
	gt_right = cv::Mat(image_width, image_height, CV_64F, temp_right.data()).t();
}

cv::Mat CalibrationAndImagesDataset::getLeftImage(const std::string& image_path)
{
	image_left = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
	CV_Assert(!image_left.empty());
	CV_Assert(image_left.type() == CV_8U);
	return image_left;
}

cv::Mat CalibrationAndImagesDataset::getRightImage(const std::string& image_path)
{
	image_right = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
	CV_Assert(!image_right.empty());
	CV_Assert(image_right.type() == CV_8U);
	return image_right;
}

cv::Mat CalibrationAndImagesDataset::getLeftGT()
{
	return gt_left;
}

cv::Mat CalibrationAndImagesDataset::getRightGT()
{
	return gt_right;
}

cv::Mat CalibrationAndImagesDataset::getLeftCamMatrix()
{
	return cam_0.t();
}

cv::Mat CalibrationAndImagesDataset::getRightCamMatrix()
{
	return cam_1.t();
}

double CalibrationAndImagesDataset::getDoffs()
{
	return disparity_offset;
}

double CalibrationAndImagesDataset::getBaseline()
{
	return camera_baseline;
}

int CalibrationAndImagesDataset::getWidth()
{
	return image_width;
}

int CalibrationAndImagesDataset::getHeight()
{
	return image_height;
}

bool isEdgeSimulatedGrid(const int& TL, const int& TR, const int& BL, const int& BR, const cv::Mat& complete_data_matrix, EdgeDirection& edge_direction, const double& threshold_disp_dist)
{
	double pivot_disparity = complete_data_matrix.at<double>(TL, 5);
	bool is_edge = false;
	if (abs(complete_data_matrix.at<double>(TL, 5) - complete_data_matrix.at<double>(TR, 5)) > threshold_disp_dist) is_edge = true;
	if (abs(complete_data_matrix.at<double>(TL, 5) - complete_data_matrix.at<double>(BL, 5)) > threshold_disp_dist) is_edge = true;
	if (abs(complete_data_matrix.at<double>(TL, 5) - complete_data_matrix.at<double>(BR, 5)) > threshold_disp_dist) is_edge = true;
	if (abs(complete_data_matrix.at<double>(BR, 5) - complete_data_matrix.at<double>(TR, 5)) > threshold_disp_dist) is_edge = true;
	if (abs(complete_data_matrix.at<double>(BR, 5) - complete_data_matrix.at<double>(BL, 5)) > threshold_disp_dist) is_edge = true;
	if (abs(complete_data_matrix.at<double>(TR, 5) - complete_data_matrix.at<double>(BL, 5)) > threshold_disp_dist) is_edge = true;

	if (is_edge) {
		edge_direction = EdgeDirection::undefined;
		if (abs(complete_data_matrix.at<double>(TL, 5) - complete_data_matrix.at<double>(TR, 5)) > threshold_disp_dist && 
			abs(complete_data_matrix.at<double>(BL, 5) - complete_data_matrix.at<double>(BR, 5)) > threshold_disp_dist) edge_direction = EdgeDirection::vertical;
		if (abs(complete_data_matrix.at<double>(TL, 5) - complete_data_matrix.at<double>(BL, 5)) > threshold_disp_dist && 
			abs(complete_data_matrix.at<double>(TR, 5) - complete_data_matrix.at<double>(BR, 5)) > threshold_disp_dist) edge_direction = EdgeDirection::horizontal;
	}
	return is_edge;
}
