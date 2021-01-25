#pragma once

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/mat.hpp>

#include "common_structures.h"

class CalibrationAndImagesDataset
{
public:
	// Setter functions
	void setLeftRightCalibrationMatrices(const std::string& cam_0_, const std::string& cam_1_, const char& delete_char);
	void setDoffs(const std::string& doffs_);
	void setBaseline(const std::string& baseline_);
	void setWidth(const std::string& width_);
	void setHeight(const std::string& height_);
	void setLeftRightGroundTruth(const std::string& ground_truth_0, const std::string& ground_truth_1);

	// Getter functions
	cv::Mat getLeftImage(const std::string& image_path);
	cv::Mat getRightImage(const std::string& image_path);
	cv::Mat getLeftGT();
	cv::Mat getRightGT();
	cv::Mat getLeftCamMatrix();
	cv::Mat getRightCamMatrix();
	double getDoffs();
	double getBaseline();
	int getWidth();
	int getHeight();


private:
	cv::Mat image_left;
	cv::Mat image_right;
	cv::Mat gt_left;
	cv::Mat gt_right;
	cv::Mat cam_0;
	cv::Mat cam_1;
	double disparity_offset{ 0.0 };
	double camera_baseline{ 0.0 };
	int image_width{ 0 };
	int image_height{ 0 };
};



class GetWrongPoints {
public:
	int rows;
	cv::Mat getWrongPoints(const cv::Mat& input_cloud) {
		rows = input_cloud.rows;
		cv::Mat bad_depths = cv::Mat(rows, 1, CV_32S);

		for (size_t n = 0; n < input_cloud.rows; n++)
		{
			if (input_cloud.at<double>(n, 2) > 6000.0)
			{
				// bad points
				bad_depths.at<int>(n, 0) = 1;
			}
			else
			{
				// good points
				bad_depths.at<int>(n, 0) = 0;
			}
		}

		return bad_depths;
	}
};

struct TotalDerivatives
{
	ObservationData der_top;
	ObservationData der_bot;
	ObservationData der_left;
	ObservationData der_right;
	/*
	void show() {
		std::cout << " | from top " << der_top << 
			" | from bot " << der_bot << 
			" | from left " << der_left << 
			" | from right " << der_right << std::endl;

	}
	*/
};

enum class EdgeDirection { undefined, 
	horizontal, 
	vertical };
bool isEdgeSimulatedGrid(const int& TL, const int& TR,
	const int& BL, const int& BR,
	const cv::Mat& complete_data_matrix,
	EdgeDirection& edge_direction,
	const double& threshold_disp_dist = 15.0);

