#include "io_functions.h"

std::vector <float> io::readBinaryToFloat(const std::string filename) {

	std::vector<float> value_array;
	// Read the binary
	std::ifstream inStream(filename, std::ios::binary);
	float test;

	while (inStream.read(reinterpret_cast<char*>(&test), sizeof(float))) {
		value_array.push_back(test);
	}

	return value_array;
}

std::vector<double> io::readBinaryToDouble(const std::string filename)
{
	std::vector<double> value_array;
	// Read the binary
	std::ifstream in_stream(filename, std::ios::binary);
	double test;
	while (in_stream.read(reinterpret_cast<char*>(&test), sizeof(double))) {
		value_array.push_back(test);
	}
	
	return value_array;
}

void io::readMiddleburyCalibrationData(const std::string filename, std::vector<std::string>& calibration_data)
{
	std::string line;
	char c = '=';
	std::ifstream inputStream(filename);
	if (inputStream.is_open())
	{
		int lines_number = 0;
		while (lines_number < 6)
		{
			std::getline(inputStream, line);
			std::size_t pos_found = line.find(c);
			calibration_data.push_back(line.substr(pos_found + 1, line.length() - 1));
			++lines_number;
		}
	}
	else
	{
		std::cout << "Can't open the calibration file" << std::endl;
	}
	inputStream.close();
}

void io::readLadimoCalibration(const std::string filename, std::vector<std::string>& calibration_data)
{
	std::string line;
	char c = ':';

	std::ifstream inputStream(filename);
	if (inputStream.is_open())
	{
		while (std::getline(inputStream, line))
		{
			if (!line.empty())
			{
				std::size_t pos_found = line.find(c);
				if (pos_found != SIZE_MAX)
				{
					calibration_data.push_back(line.substr(pos_found + 2, line.length() - 1));
				}
				else
				{
					continue;
				}
			}
		}
	}
	else
	{
		std::cout << "Can't open the calibration file" << std::endl;
	}
	inputStream.close();
}
cv::Mat io::readCSV(const std::string filename, char delimiter)
{
	cv::Mat values_matrix;
	std::string line = "";
	int counter = 0;

	std::ifstream csvFile(filename);
	if (csvFile.is_open())
	{
		while (std::getline(csvFile, line))
		{
			std::stringstream sep_str(line);
			std::string vec;
			counter++;

			while (std::getline(sep_str, vec, delimiter))
			{
				values_matrix.push_back(std::stod(vec));
			}
		}
	}
	else
	{
		std::cout << "Not possible to open the file !" << std::endl;
	}
	csvFile.close();
	return values_matrix.reshape(1, counter);
}

void io::writeMatrixToFile(cv::Mat inputMatrix, std::string filename, std::string delimiter, int space)
{
	std::ofstream output_file;
	output_file.open(filename);
	int T = inputMatrix.depth();
	if (T == 1 || T == 3 || T == 4)
	{
		for (size_t i = 0; i < inputMatrix.rows; i++)
		{
			for (size_t j = 0; j < inputMatrix.cols; j++)
			{
				output_file << std::left << std::setw(space) << inputMatrix.at<int>(i, j) << delimiter;
			}
			output_file << std::endl;
		}
		output_file.close();
	}

	else if(T == 5)
	{
		for (size_t i = 0; i < inputMatrix.rows; i++)
		{
			for (size_t j = 0; j < inputMatrix.cols; j++)
			{
				output_file << std::left << std::setw(space) << inputMatrix.at<float>(i, j) << delimiter;
			}
			output_file << std::endl;
		}
		output_file.close();
	}
	else if(T == 6)
	{
		for (size_t i = 0; i < inputMatrix.rows; i++)
		{
			for (size_t j = 0; j < inputMatrix.cols; j++)
			{
				output_file << std::left << std::setw(space) << inputMatrix.at<double>(i, j) << delimiter;
			}
			output_file << std::endl;
		}
		output_file.close();
	}
}

void io::mergeFiles(std::string file_1, std::string file_2, std::string new_file, std::string delimiter)
{
	std::ofstream output_file;
	output_file.open(new_file);
	if (output_file.is_open())
	{
		std::ifstream infile_1;
		infile_1.open(file_1);

		std::ifstream infile_2;
		infile_2.open(file_2);

		std::string line_1;
		std::string line_2;

		while (std::getline(infile_1, line_1) && std::getline(infile_2, line_2)) {
			output_file << line_1 << delimiter << line_2 << std::endl;
		}

		infile_1.close();
		infile_2.close();
	}
	
	else
	{
		std::cout << "Not possible to open the file" << std::endl;
	}

	output_file.close();
}

void io::writeVectorToFile(std::vector<float> inputVector, std::string filename, std::string delimiter)
{
	std::ofstream output_file;
	output_file.open(filename);
	for (size_t i = 0; i < inputVector.size(); i++)
	{
		output_file << inputVector[i] << std::endl;
	}

	output_file.close();
}

void io::exportDisparityTo3D(std::string output_name, cv::Mat disparity_mat, double baseline, double cx, double cy, double f, double doffs)
{

	/*cam0=[3997.684 0 1176.728; 0 3997.684 1011.728; 0 0 1]
cam1=[3997.684 0 1307.839; 0 3997.684 1011.728; 0 0 1]
doffs=131.111
baseline=193.001
width=2964
height=1988
ndisp=280
isint=0
vmin=31
vmax=257
dyavg=0.918
dymax=1.516
The calibration files p*/

	// Z = baseline * f / (d + doffs)

	std::ofstream output;
	output.open(output_name);
	for (int i = 0; i < disparity_mat.rows; i++) {
		for (int j = 0; j < disparity_mat.cols; j++) {
			float Z = baseline * f / (disparity_mat.at<float>(i, j) + doffs );
			float X = ((j * 25 / 5) - cx) / f * Z;
			float Y = ((i * 25 / 5) - cy) / f * Z;
			output << X << " " << Y << " " << Z << std::endl;
		}
	}
	output.close();
}






