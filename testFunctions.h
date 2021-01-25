#ifndef TESTFUNCTIONS
#define TESTFUNCTIONS

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

#include "usedStructnFunctions.h"
#include "smallSupportFunctions.h"
#include "onGridOperations.h"

/*
void estimatedDisparityTest(const cv::Mat& lookup_table, const cv::Mat& XYZd_data, const std::vector<TotalDerivatives>& derivative_vector, const int& index_row, const int& index_col, const int sampl_fact);

void guessedDisparityTest(int image_row, int image_col, const cv::Mat& ground_truth, const cv::Mat& guessed_disparity, int& sampl_factor, int& gap);

void sgmTest(const cv::Mat& lookupTable, const cv::Mat& xyd_data, const cv::Mat& XYZ_data, const cv::Mat& left_image, const cv::Mat& right_image, const int& gap, const int& lookup_r, const int& lookup_c, std::string& path_test_matrices);
*/

// Part for testing over the real LaDimo grid
class RealGridOperationsTesting
{
public:
	// setter functions
	void setEstimationParameters(const cv::Mat& camera_matrix_left_, const cv::Mat& camera_matrix_right_, const float& baseline_, const int& sampling_factor_);
	void setNecessaryVectors(const std::vector<GridSquare>& gridSquaresVector, const std::vector<ObservationData>& observationsVector_, const std::vector<CompleteValuesDerivatives>& internalCompleteDerivativesVector_, const std::vector<CompleteValuesDerivatives>& esternalCompleteDerivativesVector_);
	void initilizeEstimationMatrices();
	void setEstimations(const cv::Mat& image_left, const cv::Mat& image_right);
	void setGuesses();
	void setSGMCostCubes(const std::vector<cv::Mat>& SGMCostCubesVector_);
	void setSGMBasedEstimations(const cv::Mat& image_left, const cv::Mat& image_right);

	// Getter
	cv::Mat getEstimations_disp();
	cv::Mat getEstimations_X();
	cv::Mat getEstimations_Y();
	cv::Mat getEstimations_Z();

	cv::Mat getEstimations_disp_no_edge();
	cv::Mat getEstimations_X_no_edge();
	cv::Mat getEstimations_Y_no_edge();
	cv::Mat getEstimations_Z_no_edge();

	cv::Mat getEstimations_disp_strong_edge();
	cv::Mat getEstimations_X_strong_edge();
	cv::Mat getEstimations_Y_strong_edge();
	cv::Mat getEstimations_Z_strong_edge();

	cv::Mat getEstimations_disp_soft_edge();
	cv::Mat getEstimations_X_soft_edge();
	cv::Mat getEstimations_Y_soft_edge();
	cv::Mat getEstimations_Z_soft_edge();

	std::vector<float> getSGMEstimations_disp();
	std::vector<float> getSGMEstimations_X();
	std::vector<float> getSGMEstimations_Y();
	std::vector<float> getSGMEstimations_Z();
	std::vector<float> getSGMEstimations_x_px();
	std::vector<float> getSGMEstimations_y_px();

private:
	// Vectors to use
	std::vector<GridSquare> gridSquaresVector{};
	std::vector<ObservationData> observationsVector{};
	std::vector<cv::Mat> SGMCostCubesVector{};
	std::vector<CompleteValuesDerivatives> internalCompleteDerivativesVector{};
	std::vector<CompleteValuesDerivatives> esternalCompleteDerivativesVector{};
	//std::vector<TotalDerivatives> derivativesVector{};
	//std::vector<cv::Point2f> estimated_left_pixel_positions{};

	// Estimation computation related vectors
	std::vector<float> raw_differences{0.f, 0.f, 0.f, 0.f};
	ObservationData final_estimastion_no_edge_internal_deriv{};

	float no_edge_disp = 0.f;
	float no_edge_Z = 0.f;

	int avg_square_size_4eye = 27;
	int avg_square_size_micro = 17;


	// Internal Parameters
	cv::Mat left_image_limits{};
	cv::Mat right_image_limits{};
	int sampling_factor{ 4 };
	float step{ 0.25f };
	float baseline{74.f};
	int window_size = 7;
	cv::Mat camera_matrix_left{};
	cv::Mat camera_matrix_right{};

	float P0{ 0.f };
	float P1{ 0.f };
	float P2{ 0.f };
	float P3{ 0.f };
	float P4{ 0.f }; 

	float P1_sgm{ 0.f };
	float P2_sgm{ 0.f };

	// Final Complete Functions
	// Final estimation matrices
	cv::Mat estimated_Disparity;
	cv::Mat estimated_X;
	cv::Mat estimated_Y;
	cv::Mat estimated_Z;

	// Final guessed matrices
	cv::Mat guessed_Disparity;
	cv::Mat guessed_X;
	cv::Mat guessed_Y;
	cv::Mat guessed_Z;

	// Final estimation matrices
	cv::Mat estimated_Disparity_no_edge;
	cv::Mat estimated_X_no_edge;
	cv::Mat estimated_Y_no_edge;
	cv::Mat estimated_Z_no_edge;

	cv::Mat estimated_Disparity_strong_edge;
	cv::Mat estimated_X_strong_edge;
	cv::Mat estimated_Y_strong_edge;
	cv::Mat estimated_Z_strong_edge;

	cv::Mat estimated_Disparity_soft_edge;
	cv::Mat estimated_X_soft_edge;
	cv::Mat estimated_Y_soft_edge;
	cv::Mat estimated_Z_soft_edge;

	// Final guessed matrices
	cv::Mat guessed_Disparity_no_edge;
	cv::Mat guessed_X_no_edge;
	cv::Mat guessed_Y_no_edge;
	cv::Mat guessed_Z_no_edge;

	cv::Mat guessed_Disparity_strong_edge;
	cv::Mat guessed_X_strong_edge;
	cv::Mat guessed_Y_strong_edge;
	cv::Mat guessed_Z_strong_edge;

	cv::Mat guessed_Disparity_soft_edge;
	cv::Mat guessed_X_soft_edge;
	cv::Mat guessed_Y_soft_edge;
	cv::Mat guessed_Z_soft_edge;
	

	// SGM Based method vector 4 estimated values
	std::vector<float> SGMEstimatedDisparity{};
	std::vector<float> SGMEstimated_X{};
	std::vector<float> SGMEstimated_Y{};
	std::vector<float> SGMEstimated_Z{};
	std::vector<float> SGMEstimated_x_px{};
	std::vector<float> SGMEstimated_y_px{};

	// Internal methods
	struct EdgeThresholds
	{
		float light_depth_threshold = 0.15f;
		float strong_depth_threshold = 0.07f;
		float general_depth_threshold = 0.09f;
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

	enum class SGMEdgeDirection
	{
		vertical,
		horizontal,
		undefined
	};

	bool findStrongEdges(const int& TL, const int& TR, const int& BL, const int& BR, EdgeShape& edge_shape);
	bool findSoftEdges(const int& TL, const int& TR, const int& BL, const int& BR, EdgeShape& edge_shape);
	bool findSGMEdgeDirection(const int& TL, const int& TR, const int& BL, const int& BR, SGMEdgeDirection& edge_direction);

	void bilinearInterpolation(const int& TL, const int& TR, const int& BL, const int& BR, cv::Vec4f& sub_square_estimation);
	void noEdgesEstimations(const int& TL, const int& TR, const int& BL, const int& BR, const float& indx_r, const float& indx_c, std::vector<ObservationData>& no_edge_est_vec);
	void strongEdgesEstimations(const int& TL, const int& TR, const int& BL, const int& BR, const float& indx_r, const float& indx_c, const cv::Mat& im_left, const cv::Mat& im_right, std::vector<ObservationData>& strong_edge_est_vec);
	void softEdgesEstimations(const int& TL, const int& TR, const int& BL, const int& BR, const float& indx_r, const float& indx_c, const cv::Mat& im_left, const cv::Mat& im_right, std::vector<ObservationData>& soft_edge_est_vec);


	void fillSGMCostCube(const int& TL, const int& TR, const int& BL, const int& BR, const int& cost_cube_indx, const cv::Mat& image_left, const cv::Mat& image_right, const std::vector<float>& patch_disparity_values_);
	void noEdgesSGMEstimations(const int& TL, const int& TR, const int& BL, const int& BR, const int& cost_cube_indx, const std::vector<float>& patch_disparity_values_, int& vector_indx);
	void verticalEdgesSGMEstimations(const int& TL, const int& TR, const int& BL, const int& BR, const int& cost_cube_indx, const std::vector<float>& patch_disparity_values_, int& vector_indx);
	void horizontalEdgesSGMEstimations(const int& TL, const int& TR, const int& BL, const int& BR, const int& cost_cube_indx, const std::vector<float>& patch_disparity_values_, int& vector_indx);
	void undefinedEdgesSGMEstimations(const int& TL, const int& TR, const int& BL, const int& BR, const int& cost_cube_indx, const std::vector<float>& patch_disparity_values_, int& vector_indx);

	void aggregationCostDirection_0(const int& cost_cube_indx, cv::Mat& direction_cost_cube_0);
	void aggregationCostDirection_2(const int& cost_cube_indx, cv::Mat& direction_cost_cube_2);
	void aggregationCostDirection_1(const int& cost_cube_indx, cv::Mat& direction_cost_cube_1);
	void aggregationCostDirection_3(const int& cost_cube_indx, cv::Mat& direction_cost_cube_3);
};


#endif // !TESTFUNCTIONS