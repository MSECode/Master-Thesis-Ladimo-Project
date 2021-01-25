#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <typeinfo>
#include <iomanip> 

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

class io {
public:
	/**
	 * read a binary file
	 * to a vector of float
	 */
	static std::vector <float> readBinaryToFloat(const std::string filename);
	/**
	 * read a binary file
	 * to a vector of double
	 */
	static std::vector<double> readBinaryToDouble(const std::string filename);


	static cv::Mat readCSV(const std::string filename, char delimiter = ',');
	

	static void writeMatrixToFile(cv::Mat inputMatrix, std::string filename, std::string delimiter = " ", int space = 15);

	static void mergeFiles(std::string file_1, std::string file_2, std::string new_file, std::string delimiter = " ");

	static void writeVectorToFile(std::vector<float> inputVector, std::string filename, std::string delimiter = " ");

	static void exportDisparityTo3D(std::string output_name, cv::Mat disparity_mat, double baseline, double cx, double cy, double f, double doffs);

	/**
	 * Read calibration file 
	 * for the Middlebury 2014 dataset images
	 * data are stored in a vector of string in the following order:
	 * cam_0
	 * cam_1
	 * doofs
	 * baseline
	 * width
	 * height
	*/
	static void readMiddleburyCalibrationData(const std::string filename, std::vector<std::string>& calibration_data);

	/**
	 * Read calibration file 
	 * for the LaDiMo set of images
	 * data are stored in a vector of string in the following order:
	 * width
	 * height
	 * focal lenght
	 * principal point x 
	 * principal point y
	 * baseline
	 */
	static void readLadimoCalibration(const std::string filename, std::vector<std::string>& calibration_data);
};



