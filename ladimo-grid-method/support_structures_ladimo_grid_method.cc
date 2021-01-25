#include "support_structures_ladimo_grid_method.h"

void LadimoCalibrationandMatrices::setLeftRightCameraMatrices(const std::string& principal_point_left_, 
	const std::string& principal_point_right_, 
	const std::string& focal_length_left_, 
	const std::string& focal_length_right_, 
	const char& mid_char)
{
	size_t pos_mid_l = principal_point_left_.find(mid_char),
		pos_mid_r = principal_point_right_.find(mid_char);

	std::string cx_l = principal_point_left_.substr(1, pos_mid_l - 1),
		cy_l = principal_point_right_.substr(pos_mid_l + 2, principal_point_left_.length() - 2);
	principal_point_left[0] = std::stod(cx_l);
	principal_point_left[1] = std::stod(cy_l);

	std::string cx_r = principal_point_right_.substr(1, pos_mid_r - 1),
		cy_r = principal_point_right_.substr(pos_mid_r + 2, principal_point_right_.length() - 2);
	principal_point_right[0] = std::stod(cx_r);
	principal_point_right[1] = std::stod(cy_r);

	pos_mid_l = focal_length_left_.find(mid_char);
	pos_mid_r = focal_length_right_.find(mid_char);

	std::string f_l = focal_length_left_.substr(1, pos_mid_l - 1);
	focal_length_left = std::stod(f_l);

	std::string f_r = focal_length_right_.substr(1, pos_mid_l - 1);
	focal_length_right = std::stod(f_r);

	std::vector<double> cam_mat_vec_l = {
		focal_length_left, 0.0, principal_point_left[0],
			0.0, focal_length_left, principal_point_left[1],
			0.0, 0.0, 1.0
	};
	std::vector<double> cam_mat_vec_r = {
		focal_length_left, 0.0, principal_point_left[0],
			0.0, focal_length_left, principal_point_left[1],
			0.0, 0.0, 1.0
	};

	camera_matrix_left = cv::Mat(3, 3, CV_64F, cam_mat_vec_l.data()).t();
	camera_matrix_right = cv::Mat(3, 3, CV_64F, cam_mat_vec_r.data()).t();
}

void LadimoCalibrationandMatrices::setBaseline(const std::string& baseline_)
{
	baseline = std::stod(baseline_);
}

void LadimoCalibrationandMatrices::setImageDimension(const std::string& image_dim, const char& mid_char)
{
	size_t pos_mid = image_dim.find(mid_char);
	std::string width_ = image_dim.substr(1, pos_mid - 1),
		height_ = image_dim.substr(pos_mid + 2, image_dim.length() - 2);

	image_width = std::stoi(width_);
	image_height = std::stoi(height_);
}

void LadimoCalibrationandMatrices::setLeftRightTransformDeviceToStereo(const std::string& file_path_l, const std::string& file_path_r, const std::string& file_tag_l, const std::string& file_tag_r)
{
	cv::FileStorage file_transf_l(file_path_l, cv::FileStorage::READ);
	file_transf_l[file_tag_l] >> transf_dist_to_rect_l;
	file_transf_l.release();

	cv::FileStorage file_transf_r(file_path_r, cv::FileStorage::READ);
	file_transf_r[file_tag_r] >> transf_dist_to_rect_r;
	file_transf_r.release();

}

cv::Mat LadimoCalibrationandMatrices::getImageLeft(const std::string& image_path)
{
	cv::Mat image_left = cv::imread(image_path, cv::IMREAD_UNCHANGED);
	CV_Assert(!image_left.empty());
	return image_left;
}

cv::Mat LadimoCalibrationandMatrices::getImageRight(const std::string& image_path)
{
	cv::Mat image_right = cv::imread(image_path, cv::IMREAD_UNCHANGED);
	CV_Assert(!image_right.empty());
	return image_right;
}
