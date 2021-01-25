#pragma once

#include <iostream>
#include <assert.h>

#include "opencv2/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"

#include "common_structures.h"


class LadimoCalibrationandMatrices {

public:
	void setLeftRightCameraMatrices(const std::string& principal_point_left_,
		const std::string& principal_point_right_,
		const std::string& focal_length_left_,
		const std::string& focal_length_right_, 
		const char& mid_char);
	void setBaseline(const std::string& baseline_);
	void setImageDimension(const std::string& image_dim, const char& mid_char);
	void setLeftRightTransformDeviceToStereo(const std::string& file_path_l, 
		const std::string& file_path_r, 
		const std::string& file_tag_l,
		const std::string& file_tag_r);

	cv::Mat getImageLeft(const std::string& image_path);
	cv::Mat getImageRight(const std::string& image_path);

	cv::Mat getLeftCameraMatrix() {
		return camera_matrix_left.t();
	};
	cv::Mat getRightCameraMatrix() {
		return camera_matrix_right.t();
	};
	cv::Mat getLeftTransfDeviceToStereo() {
		return transf_dist_to_rect_l;
	};
	cv::Mat getRightTransfDeviceToStereo() {
		return transf_dist_to_rect_r;
	};
	double getBaseline() {
		return baseline;
	};
	int getImageWidth() {
		return image_width;
	};
	int getImageHeight() {
		return image_height;
	};

private:
	double baseline = 0.0;
	double principal_point_left[2]{}, principal_point_right[2]{};
	double focal_length_left = 0.0, focal_length_right = 0.0;
	int image_width = 0, image_height = 0;

	cv::Mat camera_matrix_left,
		camera_matrix_right,
		transf_dist_to_rect_l,
		transf_dist_to_rect_r;
};


struct GridPositions
{
	GridPositions() {};
	GridPositions(int x_, int y_) {
		x = x_;
		y = y_;
	};

	int x = 0;
	int y = 0;
	bool equals(const GridPositions& gp) {
		return (x == gp.x && y == gp.y);
	}
	GridPositions left() {
		return	GridPositions(x - 1, y);
	}
	GridPositions right() {
		return	GridPositions(x + 1, y);
	}
	GridPositions top() {
		return	GridPositions(x, y - 1);
	}
	GridPositions bottom() {
		return	GridPositions(x, y + 1);
	}
};

struct GridSquare
{
	int top_left_index{ -1 };
	int top_right_index{ -1 };
	int bottom_left_index{ -1 };
	int bottom_right_index{ -1 };
};

struct NeighbourIndexes
{
	int top{ -1 };
	int bot{ -1 };
	int left{ -1 };
	int right{ -1 };

	bool isFull()
	{
		bool is_full = false;
		if (top != -1 && left != -1 && right != -1 && bot != -1) is_full = true;
		return is_full;
	};
};

struct LaserGrid
{
	std::vector<GridPositions> grid;
};

struct CompleteDerivatives
{
	ObservationData top_deriv{};
	ObservationData bottom_deriv{};
	ObservationData left_deriv{};
	ObservationData right_deriv{};
};

struct EdgeThresholds
{
	double light_depth_threshold = 0.20;
	double strong_depth_threshold = 0.07;
	double general_depth_threshold = 0.09;
};

enum class EdgeShape
{
	pure_vertical,
	pure_horizontal,
	diagonal_top_left,
	diagonal_top_right,
	diagonal_bottom_left,
	diagonal_bottom_right,
	undefined
};
