#include "Ladimo_Stereo_Matcher.h"
#include <assert.h>

void Ladimo_Stereo_Matcher::setLaserGrid(std::vector<GridPos> grid)
{
	laser_grid.grid = grid;
	CreateNeighborIndices();
}

void Ladimo_Stereo_Matcher::setSamplingfactor(int factor_)
{
	sampling_factor = factor_;
}

void Ladimo_Stereo_Matcher::setCameraMatrices(cv::Mat cam_mat1_, cv::Mat cam_mat2_)
{
	cam_mat1 = cam_mat1_;
	f1 = cam_mat1.at<float>(0, 0);
	cx1 = cam_mat1.at<float>(0, 2);
	cy1 = cam_mat1.at<float>(1, 2);

	cam_mat2 = cam_mat2_;
	f2 = cam_mat2.at<float>(0, 0);
	cx2 = cam_mat2.at<float>(0, 2);
	cy2 = cam_mat2.at<float>(1, 2);
}

void Ladimo_Stereo_Matcher::setBaseline(float baseline_)
{
	baseline = baseline_;
}

void Ladimo_Stereo_Matcher::setDerivativesZThreshold(float threshold_)
{
	derivative_z_threshold = threshold_;
}

void Ladimo_Stereo_Matcher::setGridAngle(float angle_)
{
	grid_angle = angle_;
}

void Ladimo_Stereo_Matcher::setAvgPointDistancePxc(float avg_pointdistance_pxc_)
{
	avg_point_distance_pxc = avg_pointdistance_pxc_;
}

// observedDots vector has to have all dots, even no datam and the points need to be ordered the same way as grid poses
void Ladimo_Stereo_Matcher::run(const cv::Mat& image1, const cv::Mat& image2, std::vector<cv::Point3f>& observedDots)
{
	assert(observedDots.size() == neighbor_indices.size());
	observations = observedDots; // pointers could be faster..

	// calculate normals || UNSURE IF THIS WILL BE USED
	CalculateNormals();

	// fill data gaps when reasonable
	FillDataGaps();

}

void Ladimo_Stereo_Matcher::CreateNeighborIndices()
{
	std::vector<NeighborIndices>;
	for (size_t i = 0; i < laser_grid.grid.size(); i++) {
		NeighborIndices ni;
		GridPos gp_i = laser_grid.grid[i];
		GridPos gp_left = gp_i.left();
		GridPos gp_right = gp_i.right();
		GridPos gp_top = gp_i.top();
		GridPos gp_bottom = gp_i.bottom();

		// set neighbor indices
		for (size_t j = 0; j < laser_grid.grid.size(); j++) {
			if (laser_grid.grid[j].equals(gp_left)) ni.left = j;
			if (laser_grid.grid[j].equals(gp_right)) ni.right = j;
			if (laser_grid.grid[j].equals(gp_top)) ni.top = j;
			if (laser_grid.grid[j].equals(gp_bottom)) ni.bottom = j;
		}
	}

	// resize derivatives
	derivatives.resize(laser_grid.grid.size());
	normals.resize(laser_grid.grid.size());
}
void Ladimo_Stereo_Matcher::FillDataGaps()
{
	// TODO improve gap filling
	for (size_t i = 0; i < observations.size(); i++) {
		// no data points have 0 values
		if (!observations[i].z > 0.f) {
			// no data, check surroundings
			
			// first simple implementation, interpolate a dot from neighbors, will work decently for some cases
			if (observations[neighbor_indices[i].left].z > 0.f && observations[neighbor_indices[i].right].z > 0.f) {
				// check if points are close, dZ < limit
				if (abs(observations[neighbor_indices[i].left].z - observations[neighbor_indices[i].right].z) < 120.f) {
					observations[i] = (observations[neighbor_indices[i].left] + observations[neighbor_indices[i].left]) * 0.5f;
				}
				// else
			}
			else if( observations[neighbor_indices[i].top].z > 0.f && observations[neighbor_indices[i].bottom].z > 0.f) {
				observations[i] = (observations[neighbor_indices[i].bottom] + observations[neighbor_indices[i].bottom]) * 0.5f;
			}
			// else ?? 
			//else if()
		}
	}
}

// use neighbor indices to calculate directed derivatives
void Ladimo_Stereo_Matcher::CalculateDerivatives()
{
	// normalized standard derivatives from parameters;
	float angle_ = grid_angle / 180.f * M_PI;
	float x_ = cos(angle_) * avg_point_distance_pxc - sin(angle_) * avg_point_distance_pxc;
	float y_ = sin(angle_) * avg_point_distance_pxc + cos(angle_) * avg_point_distance_pxc;
	cv::Point2f grid_x_standard_deriv( x_, y_ );
	cv::Point2f grid_y_standard_deriv(-y_, x_);
	

	for (size_t i = 0; i < derivatives.size(); i++) {
		if (neighbor_indices[i].left > -1) {
			derivatives[i].from_left = observations[i] - observations[neighbor_indices[i].left];
		}
		if (neighbor_indices[i].top > -1) {
			derivatives[i].from_top = observations[i] - observations[neighbor_indices[i].top];
		}
		if (neighbor_indices[i].right > -1) {
			derivatives[i].from_right = observations[i] - observations[neighbor_indices[i].right];
		}
		if (neighbor_indices[i].bottom > -1) {
			derivatives[i].from_bottom = observations[i] - observations[neighbor_indices[i].bottom];
		}

		// sanity check
		if (derivatives[i].from_left.z > derivative_z_threshold) {
			// check opposite direction, if good use that
			if (derivatives[i].from_right.z > derivative_z_threshold) {
				derivatives[i].from_left = -derivatives[i].from_right;
			}
			else {
				// use defauld derivative scaled by Z
				derivatives[i].from_left = cv::Point3f(grid_x_standard_deriv.x * observations[i].z,
					                                   grid_x_standard_deriv.y * observations[i].z,
					                                   0.f);
			}
		}

		if (derivatives[i].from_right.z > derivative_z_threshold) {
			// check opposite direction, if good use that
			if (derivatives[i].from_left.z > derivative_z_threshold) {
				derivatives[i].from_right = -derivatives[i].from_left;
			}
			else {
				// use defauld derivative scaled by Z
				derivatives[i].from_right = cv::Point3f(-grid_x_standard_deriv.x * observations[i].z,
					-grid_x_standard_deriv.y * observations[i].z,
					0.f);
			}
		}

		if (derivatives[i].from_top.z > derivative_z_threshold) {
			// check opposite direction, if good use that
			if (derivatives[i].from_bottom.z > derivative_z_threshold) {
				derivatives[i].from_top = -derivatives[i].from_bottom;
			}
			else {
				// use defauld derivative scaled by Z
				derivatives[i].from_top = cv::Point3f(grid_y_standard_deriv.x * observations[i].z,
					                                  grid_y_standard_deriv.y * observations[i].z,
				                                   	  0.f);
			}
		}

		if (derivatives[i].from_bottom.z > derivative_z_threshold) {
			// check opposite direction, if good use that
			if (derivatives[i].from_top.z > derivative_z_threshold) {
				derivatives[i].from_bottom = -derivatives[i].from_top;
			}
			else {
				// use defauld derivative scaled by Z
				derivatives[i].from_bottom = cv::Point3f(-grid_y_standard_deriv.x * observations[i].z,
					                                     -grid_y_standard_deriv.y * observations[i].z,
					                                     0.f);
			}
		}
	}

	
	 
}

void Ladimo_Stereo_Matcher::CalculateNormals()
{
	// pre declare cv::Point3f
	cv::Point3f normal;
	cv::Point3f n;
	cv::Point3f v1;
	cv::Point3f v2;
	for (size_t i = 0; i < normals.size(); i++) {
		normal = cv::Point3f(0.f, 0.f, 0.f);
		
		// left bottom
		if (neighbor_indices[i].left > -1 && neighbor_indices[i].bottom > -1) {
			// vectors
			v1 = observations[neighbor_indices[i].left] - observations[i];
			v2 = observations[neighbor_indices[i].bottom] - observations[i];
			// normal diraction estimate from cross product
			n = v1.cross(v2);
			// inverse distance weighting. Points wits large distances will have a large cross product -> small weight
			normal += n / (n.dot(n));
		}
		
		//bottom right
		if (neighbor_indices[i].bottom > -1 && neighbor_indices[i].right > -1) {
			v1 = observations[neighbor_indices[i].bottom] - observations[i];
			v2 = observations[neighbor_indices[i].right] - observations[i];
			n = v1.cross(v2);
			normal += n / (n.dot(n));
		}

		// right top
		if (neighbor_indices[i].right > -1 && neighbor_indices[i].top > -1) {
			v1 = observations[neighbor_indices[i].right] - observations[i];
			v2 = observations[neighbor_indices[i].top] - observations[i];
			n = v1.cross(v2);
			normal += n / (n.dot(n));
		}

		// top left
		if (neighbor_indices[i].top > -1 && neighbor_indices[i].left > -1) {
			v1 = observations[neighbor_indices[i].top] - observations[i];
			v2 = observations[neighbor_indices[i].left] - observations[i];
			n = v1.cross(v2);
			normal += n / (n.dot(n));
		}
		if (abs(normal.z) > 0.f) {
			normal /= cv::norm(normal);
		}
		normals[i] = normal;
	}

}
