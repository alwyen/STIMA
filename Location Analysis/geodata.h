#include <map>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <string>

using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::VectorXd;

// Ask Ben about multi-view geometry error reduction
// Mention the diamond shape area reduction by camera placement/more images.
// What might be the camera callibration (Alex was having issues with caltech camera callibration) - possible alternatives.
//      - not subpixel error. Possibly use opencv. 
//      - Ask if Alex can give him the images for him to try it - see if it's the images or alex's use of the software.

class GeoData {
public:
	GeoData(std::string iso_csv_path, std::string test_csv_path, std::string ground_truth_csv_path);
	struct Geo;
	struct LightData;
	struct GroundTruth;
	std::map<std::string, Geo> getGeoData();
	std::map<std::string, std::map<int, std::vector<LightData>>> getLightData(); // update data structure to map(map(vector))
	std::map<int, GroundTruth> getGtData();
private:
	std::map<std::string, Geo> geoData;
	std::map<std::string, std::map<int, std::vector<LightData>>> lightData;
	std::map<int, GroundTruth> gtData;
	void fillCameraData(std::string camera_data_csv_path);
	void fillTestData(std::string test_csv_path);
	void fillGTData(std::string ground_truth_csv_path);
};
