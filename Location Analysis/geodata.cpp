// Everything is in CSV files
// Geocentric Coordinates (x, y, z)
// Geodedic Coordinates (latitude, longitude, height)
// Yaw, Pitch, Roll coordinates (Angle in Radians) of the gimble
// Yaw, Pitch, Roll coordinates (Angle in Radians) of the platform
// Feature coordinates of the light in the image (x,y coordinates (inhomogenous))

//constructor that will take in three csv files
//constructor will fill out the structs for each csv file
//Then put each struct to be accessible from a map that is public to the container


//Functions necessary for first CSV (auto focus) file -> General function for reading and filling the data
// Functions for filling data -> matrix input for geo, yaw-pitch-roll 
//Functions necessary for second CSV (Test CSV) file -> General function for reading and filling in the data
// Data will be dict(key(image), dict(key(light), struct(dir, x, y)));

//Final CSV (ground truth):
//	Map of structs<light#, struct of data> of structs containing all the info per light

#include <map>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include "geodata.h"

using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::VectorXd;


struct GeoData::Geo {
	double Lat;
	double Long;
	double Ortho;
	double Yaw_X;
	double Pitch_Y;
	double Roll_Z;
	double Geocentric_X;
	double Geocentric_Y;
	double Geocentric_Z;
}; 

struct GeoData::LightData {
	//std::string light_num;
	std::string left_Right;
	double X_coord;
	double Y_coord;
};

struct GeoData::GroundTruth {
	double Lat;
	double Long;
	double Est_ortho;
	double Geocentric_X;
	double Geocentric_Y;
	double Geocentric_Z;
};

void GeoData::fillCameraData(std::string camera_data_csv_path) {
	std::ifstream fin;

	// Open an existing file
	fin.open(camera_data_csv_path, std::ios::in);
	std::vector<std::string> row;

	std::string line, word, temp;

	while (std::getline(fin, line)) {
		row.clear();

		// read an entire row and
		// store it in a string variable 'line'
		//getline(fin, line);
		//std::cout << line << std::endl;
		// used for breaking words
		std::stringstream s(line);

		while (std::getline(s, word, ',')) {
			//std::cout << word << std::endl;
			row.push_back(word);
		}
		if (row[2] == "Lat") {
			continue;
		}
		//std::cout << std::stod(row[2]) << std::endl;

		Geo data = {
			std::stod(row[2]),
			std::stod(row[3]),
			std::stod(row[4]),
			std::stod(row[5]),
			std::stod(row[6]),
			std::stod(row[7]),
			std::stod(row[8]),
			std::stod(row[9]),
			std::stod(row[10]),
		};
		
		geoData.insert({ row[1], data });

	}
}
void GeoData::fillTestData(std::string test_csv_path) {
	std::ifstream fin;

	// Open an existing file
	fin.open(test_csv_path, std::ios::in);
	std::vector<std::string> row;

	std::string line, word, temp;

	while (std::getline(fin, line)) {
		row.clear();

		// read an entire row and
		// store it in a string variable 'line'
		//getline(fin, line);
		//std::cout << line << std::endl;
		// used for breaking words
		std::stringstream s(line);

		while (std::getline(s, word, ',')) {
			//std::cout << word << std::endl;
			row.push_back(word);
		}
		if (row[0] == "Image_Name") {
			continue;
		}
		LightData data = {
			//row[1],
			row[2],
			std::stod(row[3]),
			std::stod(row[4])
		};
		if (lightData.find(row[0]) == lightData.end()) {
			lightData[row[0]] = std::map<int, std::vector<LightData>>();
		}
		else if (lightData[row[0]].find(std::stoi(row[1])) == lightData[row[0]].end()) {
			lightData[row[0]][stoi(row[1])] = std::vector<LightData>();
		}
		lightData[row[0]][stoi(row[1])].push_back(data);
	}
}
void GeoData::fillGTData(std::string ground_truth_csv_path) {
	std::ifstream fin;

	// Open an existing file
	fin.open(ground_truth_csv_path, std::ios::in);
	std::vector<std::string> row;

	std::string line, word, temp;

	while (std::getline(fin, line)) {
		row.clear();

		// read an entire row and
		// store it in a string variable 'line'
		//getline(fin, line);
		//std::cout << line << std::endl;
		// used for breaking words
		std::stringstream s(line);

		while (std::getline(s, word, ',')) {
			//std::cout << word << std::endl;
			row.push_back(word);
		}
		if (row[0] == "Light_Number") {
			continue;
		}

		GroundTruth data = {
			std::stod(row[1]),
			std::stod(row[2]),
			std::stod(row[3]),
			std::stod(row[4]),
			std::stod(row[5]),
			std::stod(row[6])
		};
		gtData.insert({ std::stoi(row[0]), data });
	}
}

GeoData::GeoData(std::string camera_data_csv_path, std::string test_csv_path, std::string ground_truth_csv_path) {
	fillGTData(ground_truth_csv_path);
	fillCameraData(camera_data_csv_path);
	fillTestData(test_csv_path);
}

std::map<std::string, GeoData::Geo> GeoData::getGeoData() {
	return geoData;
}
std::map<std::string, std::map<int, std::vector<GeoData::LightData>>> GeoData::getLightData() {
	return lightData;
}
std::map<int, GeoData::GroundTruth> GeoData::getGtData() {
	return gtData;
}
//
//int main() {
//	std::string pathCameraData = "C:/Users/aquir/Downloads/1_350_exp_3200_iso_autofocus_out.csv";
//	std::string pathTest = "C:/Users/aquir/Downloads/10_08_21_test.csv";
//	std::string pathGT = "C:/Users/aquir/Downloads/ground_truth_gis_lights.csv";
//
//	GeoData gd = GeoData(pathCameraData, pathTest, pathGT);
//	std::map<std::string, GeoData::Geo> cameraData = gd.getGeoData();
//
//	std::map<std::string, std::map<int, std::vector<GeoData::LightData>>> lightData = gd.getLightData();
//
//	std::map<int, GeoData::GroundTruth> gtData = gd.getGtData();
//
//	std::cout << "ISO data" << std::endl;
//	for (auto i : cameraData) {
//		std::cout << i.first << " " << i.second.Geocentric_X << std::endl;
//	}
//
//	std::cout << "Test data" << std::endl;
//	for (auto i : lightData) {
//		std::cout << i.first << std::endl;
//		for (auto j : lightData[i.first]) {
//			std::cout << " " << j.first << " " << j.second[0].left_Right << " " << j.second[0].X_coord << std::endl;
//			std::cout << " " << j.first << " " << j.second[1].left_Right << " " << j.second[1].X_coord << std::endl;
//		}
//	}
//
//	std::cout << "GT data" << std::endl;
//	for (auto &i : gtData) {
//		std::cout << i.first << " " << i.second.Geocentric_X << std::endl;
//	}
//}