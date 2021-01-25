#include "usedStructnFunctions.h"

cv::Mat MatrixLimits::defineMatrixLimits(int top, int bottom, int left, int right)
{
	std::array<std::array<int, 2>, 2> limits_array = { {{top, bottom}, {left, right}} };

	return cv::Mat(2, 2, CV_32S, limits_array.data()).t();
}

void MatrixnCalibration::setLeftRightCalibrationMat(const std::string& cam_0_, const std::string& cam_1_, const char& delete_char)
{
	std::vector<float> tempVectorLeft;
	std::string tempSubstring = cam_0_.substr(1, cam_0_.length() - 2);
	tempSubstring.erase(std::remove(tempSubstring.begin(), tempSubstring.end(), delete_char), tempSubstring.end());
	std::istringstream inputString_l(tempSubstring);
	tempVectorLeft.assign(std::istream_iterator<float>(inputString_l), std::istream_iterator<float>());
	cam_0 = cv::Mat(3, 3, CV_32F, tempVectorLeft.data()).t();

	std::vector<float> tempVectorRight;
	tempSubstring = cam_1_.substr(1, cam_1_.length() - 2);
	tempSubstring.erase(std::remove(tempSubstring.begin(), tempSubstring.end(), delete_char), tempSubstring.end());
	std::istringstream inputString_r(tempSubstring);
	tempVectorRight.assign(std::istream_iterator<float>(inputString_r), std::istream_iterator<float>());
	cam_1 = cv::Mat(3, 3, CV_32F, tempVectorRight.data()).t();
}

void MatrixnCalibration::setDoffs(const std::string& doffs_)
{
	disparity_offset = std::stof(doffs_);
}

void MatrixnCalibration::setBaseline(const std::string& baseline_)
{
	camera_baseline = std::stof(baseline_);
}

void MatrixnCalibration::setWidth(const std::string& width_)
{
	image_width = std::stoi(width_);
}

void MatrixnCalibration::setHeight(const std::string& height_)
{
	image_height = std::stoi(height_);
}

void MatrixnCalibration::setLeftRightGroundTruth(const std::string& ground_truth_0, const std::string& ground_truth_1)
{
	std::vector<float> temp_left = io::readBinary(ground_truth_0);
	std::vector<float> temp_right = io::readBinary(ground_truth_1);

	groundTruthLeft = cv::Mat(image_width, image_height, CV_32F, temp_left.data()).t();
	groundTruthRight = cv::Mat(image_width, image_height, CV_32F, temp_right.data()).t();
}

cv::Mat MatrixnCalibration::getLeftImage(const std::string& image_path)
{
	imageLeft = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
	CV_Assert(!imageLeft.empty());
	CV_Assert(imageLeft.type() == CV_8U);
	return imageLeft;
}

cv::Mat MatrixnCalibration::getRightImage(const std::string& image_path)
{
	imageRight = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
	CV_Assert(!imageRight.empty());
	CV_Assert(imageRight.type() == CV_8U);
	return imageRight;
}

cv::Mat MatrixnCalibration::getLeftCamMatrix()
{
	return cam_0.t();
}

cv::Mat MatrixnCalibration::getRightCamMatrix()
{
	return cam_1.t();
}

float MatrixnCalibration::getDoffs()
{
	return disparity_offset;
}

float MatrixnCalibration::getBaseline()
{
	return camera_baseline;
}

int MatrixnCalibration::getWidth()
{
	return image_width;
}

int MatrixnCalibration::getHeight()
{
	return image_height;
}

cv::Mat MatrixnCalibration::getLeftGT()
{
	return groundTruthLeft;
}

cv::Mat MatrixnCalibration::getRightGT()
{
	return groundTruthRight;
}

void GridParameters::setGap(int gap_)
{
	gap = gap_;
}

void GridParameters::setSamples(int samples_)
{
	samples = samples_;
}

void GridParameters::setGridPoints(const int& im_width, const int& im_height)
{
	widthPoints = (im_width - gap) / gap;
	heightPoints = (im_height - gap) / gap;
}

int GridParameters::getGap()
{
	return gap;
}

int GridParameters::getSamples()
{
	return samples;
}

int GridParameters::getWidthPoints()
{
	return widthPoints;
}

int GridParameters::getHeightPoints()
{
	return heightPoints;
}

void LadimoGridData::setPixelData(const int& gap, const cv::Mat& left_gt, const cv::Mat& right_gt, const int& points_on_width, const int& points_on_height)
{
	cv::Mat pixelDataLeft_ = cv::Mat(points_on_width * points_on_height, 3, CV_32F);
	cv::Mat pixelDataRight_ = cv::Mat(points_on_width * points_on_height, 3, CV_32F);
	float c = 0;

	for (float i = float(gap); i < float(left_gt.rows) - float(gap) + 1; i += float(gap))
	{
		for (float j = float(gap); j < float(left_gt.cols) - float(gap) + 1; j += float(gap))
		{
			pixelDataLeft_.at<float>(c, 0) = j;
			pixelDataLeft_.at<float>(c, 1) = i;
			pixelDataLeft_.at<float>(c, 2) = left_gt.at<float>(i, j);

			pixelDataRight_.at<float>(c, 0) = pixelDataLeft_.at<float>(c, 0) - left_gt.at<float>(i, j);
			pixelDataRight_.at<float>(c, 1) = i;
			if (pixelDataRight_.at<float>(c, 0) > 0)
			{
				pixelDataRight_.at<float>(c, 2) = right_gt.at<float>(i, pixelDataRight_.at<float>(c, 0));
			}
			else
			{
				pixelDataRight_.at<float>(c, 2) = -1.f;
			}
			++c;
		}
	}
	pixelDataLeft = pixelDataLeft_;
	pixelDataRight = pixelDataRight_;
}

void LadimoGridData::setSpaceData(const float& baseline, const float& doffs, const cv::Mat& cam_0, const cv::Mat& cam_1, const int& points_on_width, const int& points_on_height)
{
	cv::Mat spaceDataLeft_ = cv::Mat(points_on_width * points_on_height, 3, CV_32F);
	cv::Mat spaceDataRight_ = cv::Mat(points_on_width * points_on_height, 3, CV_32F);

	for (size_t i = 0; i < spaceDataLeft_.rows; i++)
	{
		spaceDataLeft_.at<float>(i, 2) = (baseline * cam_0.at<float>(0, 0)) / (pixelDataLeft.at<float>(i, 2) + doffs);
		spaceDataLeft_.at<float>(i, 0) = (pixelDataLeft.at<float>(i, 0) - cam_0.at<float>(0, 2)) * spaceDataLeft_.at<float>(i, 2) / cam_0.at<float>(0, 0);
		spaceDataLeft_.at<float>(i, 1) = (pixelDataLeft.at<float>(i, 1) - cam_0.at<float>(1, 2)) * spaceDataLeft_.at<float>(i, 2) / cam_0.at<float>(1, 1);

		spaceDataRight_.at<float>(i, 2) = (baseline * cam_1.at<float>(0, 0)) / (pixelDataRight.at<float>(i, 2) + doffs);
		spaceDataRight_.at<float>(i, 0) = (pixelDataRight.at<float>(i, 0) - cam_1.at<float>(0, 2)) * spaceDataRight_.at<float>(i, 2) / cam_1.at<float>(0, 0) + baseline;
		spaceDataRight_.at<float>(i, 1) = (pixelDataRight.at<float>(i, 1) - cam_1.at<float>(1, 2)) * spaceDataRight_.at<float>(i, 2) / cam_1.at<float>(1, 1);
	}

	spaceDataLeft = spaceDataLeft_;
	spaceDataRight = spaceDataRight_;

}

cv::Mat LadimoGridData::getPixelDataLeft()
{
	return pixelDataLeft;
}

cv::Mat LadimoGridData::getPixelDataRight()
{
	return pixelDataRight;
}

cv::Mat LadimoGridData::getSpaceDataLeft()
{
	return spaceDataLeft;
}

cv::Mat LadimoGridData::getSpaceDataRight()
{
	return spaceDataRight;
}

void LookupTable::setLookupMatrix(const int& rows_number, const int& points_on_width, const int& points_on_height)
{
	std::vector<int> tempVector(rows_number);
	std::iota(std::begin(tempVector), std::end(tempVector), 0);
	
	lookupMatrix = cv::Mat(points_on_height, points_on_width, CV_32S, tempVector.data()).t();
}

cv::Mat LookupTable::getLookupMatrix()
{
	return lookupMatrix.t();
}

void Derivatives::show()
{
	std::cout << "from top " << from_top << " | from bot " << from_bot << " | from left " << from_left << " | from right " << from_right << "\n";
}

void TotalDerivatives::show()
{
	std::cout << "from top " << der_top << " | from bot " << der_bot << " | from left " << der_left << " | from right " << der_right << "\n";
}

void RealLadimoGridData::setPixelnSpaceData(const float& baseline, const cv::Mat& cam_0, const cv::Mat& cam_1, const int& gap, const cv::Mat& left_gt, const cv::Mat& right_gt, const int& points_on_width, const int& points_on_height)
{
	cv::Mat pixelDataLeft_ = cv::Mat(points_on_width * points_on_height, 3, CV_32F);
	cv::Mat pixelDataRight_ = cv::Mat(points_on_width * points_on_height, 3, CV_32F);

	cv::Mat spaceDataLeft_ = cv::Mat(points_on_width * points_on_height, 3, CV_32F);
	cv::Mat spaceDataRight_ = cv::Mat(points_on_width * points_on_height, 3, CV_32F);

	int c = 0;

	for (float i = float(gap); i < float(left_gt.rows) - float(gap) + 1; i += float(gap))
	{
		for (float j = float(gap); j < float(left_gt.cols) - float(gap) + 1; j += float(gap))
		{
			// Pixel Data Left
			pixelDataLeft_.at<float>(c, 0) = j;
			pixelDataLeft_.at<float>(c, 1) = i;
			pixelDataLeft_.at<float>(c, 2) = baseline * cam_0.at<float>(0, 0) / left_gt.at<float>(i, j);
			/*
			std::cout << "depth value: " << left_gt.at<float>(i, j) << "   " <<
				"corresponding disparity: " << pixelDataLeft_.at<float>(c, 2) << std::endl;
			*/

			// Space Data Left
			spaceDataLeft_.at<float>(c, 2) = left_gt.at<float>(i, j);
			spaceDataLeft_.at<float>(c, 0) = (pixelDataLeft_.at<float>(c, 0) - cam_0.at<float>(0, 2)) * spaceDataLeft_.at<float>(c, 2) / cam_0.at<float>(0, 0);
			spaceDataLeft_.at<float>(c, 1) = (pixelDataLeft_.at<float>(c, 1) - cam_0.at<float>(1, 2)) * spaceDataLeft_.at<float>(c, 2) / cam_0.at<float>(1, 1);
			
			/*
			std::cout << "depth value: " << spaceDataLeft_.at<float>(c, 2) << "   " <<
				"X value: " << spaceDataLeft_.at<float>(c, 0) << "   " <<
				"Y value: " << spaceDataLeft_.at<float>(c, 1) << "   " << 
				"x value: " << pixelDataLeft_.at<float>(c, 0) << "   " <<
				"y value: " << pixelDataLeft_.at<float>(c, 1) << "   " << std::endl;
			*/

			// Pixel and Space Data Right
			pixelDataRight_.at<float>(c, 0) = pixelDataLeft_.at<float>(c, 0) - pixelDataLeft_.at<float>(c, 2);
			pixelDataRight_.at<float>(c, 1) = i;
			if (pixelDataRight_.at<float>(c, 0) > 0)
			{
				pixelDataRight_.at<float>(c, 2) = baseline * cam_1.at<float>(0, 0) / right_gt.at<float>(i, round(pixelDataRight_.at<float>(c, 0)));
				spaceDataRight_.at<float>(c, 2) = right_gt.at<float>(i, round(pixelDataRight_.at<float>(c, 0)));
			}
			else
			{
				pixelDataRight_.at<float>(c, 2) = -1.f;
				spaceDataRight_.at<float>(c, 2) = 8000.f;
			}
			spaceDataRight_.at<float>(c, 0) = (pixelDataRight_.at<float>(c, 0) - cam_1.at<float>(0, 2)) * spaceDataRight_.at<float>(c, 2) / cam_1.at<float>(0, 0);
			spaceDataRight_.at<float>(c, 1) = (pixelDataRight_.at<float>(c, 1) - cam_1.at<float>(1, 2)) * spaceDataRight_.at<float>(c, 2) / cam_1.at<float>(1, 1);
			
			++c;
		}
	}
	pixelDataLeft = pixelDataLeft_;
	pixelDataRight = pixelDataRight_;

	spaceDataLeft = spaceDataLeft_;
	spaceDataRight = spaceDataRight_;
}

cv::Mat RealLadimoGridData::getPixelDataLeft()
{
	return pixelDataLeft;
}

cv::Mat RealLadimoGridData::getPixelDataRight()
{
	return pixelDataRight;
}

cv::Mat RealLadimoGridData::getSpaceDataLeft()
{
	return spaceDataLeft;
}

cv::Mat RealLadimoGridData::getSpaceDataRight()
{
	return spaceDataRight;
}

cv::Mat getExportedMatrix(const cv::Mat& inputMatrix)
{
	cv::Mat output_guessed_disparity = cv::Mat(inputMatrix.rows * inputMatrix.cols, 3, CV_32F);
	int c = 0;
	for (size_t i = 0; i < inputMatrix.rows; i++)
	{
		for (size_t j = 0; j < inputMatrix.cols; j++)
		{
			output_guessed_disparity.at<float>(c, 0) = j;
			output_guessed_disparity.at<float>(c, 1) = i;
			output_guessed_disparity.at<float>(c, 2) = inputMatrix.at<float>(i, j);

			c++;
		}
	}
	return output_guessed_disparity;
}
