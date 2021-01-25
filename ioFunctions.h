#ifndef IOFUNCTIONS
#define IOFUNCTIONS

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
	static std::vector <float> readBinary(const std::string filename);

	static cv::Mat readCSV(const std::string filename, char delimiter = ',');
	

	static void writeMatrixToFile(cv::Mat inputMatrix, std::string filename, std::string delimiter = " ", int space = 15);

	static void mergeFiles(std::string file_1, std::string file_2, std::string new_file, std::string delimiter = " ");

	static void writeVectorToFile(std::vector<float> inputVector, std::string filename, std::string delimiter = " ");

	static void exportDisparityTo3D(std::string output_name, cv::Mat disparity_mat, double baseline, double cx, double cy, double f, double doffs);

	static void readCalibrationData(const std::string filename, std::vector<std::string>& calibration_data);

	static void readLadimoCalibration(const std::string filename, std::vector<std::string>& calibration_data);
};

#endif // !IOFUNCTIONS

