#ifndef FILE_OPERATE
#define FILE_OPERATE
#include"../../single_leg_without_point_trace/single_leg.h"
#include<Eigen/Dense>
#include<fstream>
#include<string>
#include<iostream>

void save_controller_params(const std::vector<Eigen::MatrixXd>& feedback_gains,
                            const std::vector<Eigen::VectorXd>& feedforward_controls,
                            const char* file_path="new_controller_parameter"){
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", ",");
	std::ofstream file(file_path, std::ios::out);
    
    assert(feedback_gains.size()==feedforward_controls.size());
    int time_steps=feedback_gains.size();
	if (file.is_open())
	{
        for(int i=0;i<time_steps;++i){
            file << feedback_gains[i].format(CSVFormat);
            file << "\n";
            file << feedforward_controls[i].format(CSVFormat);
            file << "\n";
        }
		file.close();
	}
}

void load_controller_params(std::vector<Eigen::MatrixXd> &feedback_gains,
                           std::vector<Eigen::VectorXd> &feedforward_controls,
                           const char* file_path="/home/zhangduo/test_ws/controller_parameter"){
    std::ifstream matrixDataFile(file_path);

    std::string matrixRowString;
    std::string matrixEntry;
    int counter=0;

    while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
	{
		std::stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.    
        std::vector<double> matrixEntries;
		while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
		{
			matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
		}
        
        if(counter%2==0){
            Eigen::MatrixXd res=Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), 2, matrixEntries.size() / 2);
            feedback_gains.push_back(res);
        }
        else{
            Eigen::MatrixXd res=Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), 2, matrixEntries.size() / 2);
            feedforward_controls.push_back(res);
        }
        counter++;
	}
}

#endif