#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <Eigen/Dense>
using json = nlohmann::json;
using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;

struct KernelConfigAttribute{
	int CIN = 0; 
	int HW = 0; 
	int CIN1 = 0;
	int CIN2 = 0;
	int CIN3 = 0;
	int CIN4 = 0;
	int COUT = 0;
	int KERNEL_SIZE = 0;
	int STRIDES = 0;
	int POOL_STRIDES = 0;
	double power = 0.0; double latency = 0.0; double energy = 0.0; 
};

std::map<std::string, std::function<void(KernelConfigAttribute&, const int)>> fieldSetters = {
	{"CIN", [](KernelConfigAttribute& attr, const int val) { attr.CIN = val;}},
	{"CIN1", [](KernelConfigAttribute& attr, const int val) { attr.CIN1 = val;}},
	{"CIN2", [](KernelConfigAttribute& attr, const int val) { attr.CIN2 = val;}},
	{"CIN3", [](KernelConfigAttribute& attr, const int val) { attr.CIN3 = val;}},
	{"CIN4", [](KernelConfigAttribute& attr, const int val) { attr.CIN4 = val;}},
	{"COUT", [](KernelConfigAttribute& attr, const int val) { attr.COUT = val;}},
	{"KERNEL_SIZE", [](KernelConfigAttribute& attr, const int val) { attr.KERNEL_SIZE = val;}},
	{"STRIDES", [](KernelConfigAttribute& attr, const int val) { attr.STRIDES = val;}},
	{"POOL_STRIDES", [](KernelConfigAttribute& attr, const int val) { attr.POOL_STRIDES = val;}},
	{"HW", [](KernelConfigAttribute& attr, const int val) { attr.HW = val;}}
};



void parseInput(std::string fileName, std::unordered_map<std::string, KernelConfigAttribute> &kernel_config);
void parseKernelConfigAttribute(std::string directoryName, std::unordered_map<std::string, KernelConfigAttribute> &kernel_config, void (*parseConfig)(std::string, std::unordered_map<std::string, KernelConfigAttribute>&));
void parseLatency(std::string fileName, std::unordered_map<std::string, KernelConfigAttribute> &kernel_config);
void parseEnergy(std::string fileName, std::unordered_map<std::string, KernelConfigAttribute> &kernel_config);
void parsePower(std::string fileName, std::unordered_map<std::string, KernelConfigAttribute> &kernel_config);


void printKernelConfigAttributes(std::unordered_map<std::string, KernelConfigAttribute> &kernel_config);
void printKernelPower(std::unordered_map<std::string, KernelConfigAttribute> &kernel_config);
void printKernelEnergy(std::unordered_map<std::string, KernelConfigAttribute> &kernel_config);
void printKernelLatency(std::unordered_map<std::string, KernelConfigAttribute> &kernel_config);

Eigen::MatrixXd multiOutputRegression(int inputs, int outputs, std::unordered_map<std::string, KernelConfigAttribute> &kernel_config);
Eigen::VectorXd predict(const Eigen::MatrixXd& coefficients, double cin, double hw);
double calculateMSE(int outputs, const Eigen::MatrixXd& coefficients, std::unordered_map<std::string, KernelConfigAttribute> &kernel_config);
int main() {
	std::unordered_map<std::string, KernelConfigAttribute> kernel_config;
	parseKernelConfigAttribute("kernel_config/results/Addrelu", kernel_config, parseInput);
	parseLatency("kernel_latency/addrelu_latency.json", kernel_config);
	parsePower("kernel_power/addrelu_power.json", kernel_config);
	parseEnergy("kernel_energy/addrelu_energy.json", kernel_config);
	printKernelConfigAttributes(kernel_config);
	Eigen::MatrixXd coefficients = multiOutputRegression(2,3,kernel_config);
	std::cout << "Coefficients of model:\n" << coefficients << "\n";
	std::cout << "Current MSE: " << calculateMSE(3, coefficients, kernel_config) << "\n";
	std::cout << "Sample size: " << kernel_config.size() << "\n";
	return 0;
}


double calculateMSE(int outputs, const Eigen::MatrixXd& coefficients, std::unordered_map<std::string, KernelConfigAttribute> &kernel_config) {
	Eigen::MatrixXd Y_actual(static_cast<int>(kernel_config.size()), outputs);
	Eigen::MatrixXd Y_predicted(static_cast<int>(kernel_config.size()), outputs);
	int row {0};
	for(const auto &pair : kernel_config) {
		Y_actual(row, 0) = pair.second.power;  
	        Y_actual(row, 1) = pair.second.energy;
	        Y_actual(row, 2) = pair.second.latency;
		
	        Y_predicted.row(row) =  predict(coefficients, pair.second.CIN, pair.second.HW);
	        row++;
	}

    	Eigen::MatrixXd errors = Y_actual - Y_predicted;
	Eigen::MatrixXd squared_errors = errors.array().square();
	return squared_errors.mean();
}
Eigen::VectorXd predict(const Eigen::MatrixXd& coefficients, double cin, double hw) {
	Eigen::RowVectorXd input(3);
	input << 1, cin, hw;  
	Eigen::VectorXd predictions = input * coefficients;
	return predictions;  // [predicted_power, predicted_energy, predicted_latency]
}
Eigen::MatrixXd multiOutputRegression(int inputs, int outputs, std::unordered_map<std::string, KernelConfigAttribute> &kernel_config) {
	Eigen::MatrixXd X(static_cast<int>(kernel_config.size()), inputs + 1);

	Eigen::MatrixXd Y(static_cast<int>(kernel_config.size()), outputs);
	int row = 0;
	for(const auto &pair : kernel_config) {
	        X(row, 0) = 1;  
	        X(row, 1) = pair.second.CIN;
	        X(row, 2) = pair.second.HW;

	        Y(row, 0) = pair.second.power;
	        Y(row, 1) = pair.second.energy;
	        Y(row, 2) = pair.second.latency;
	        row++;
	}
	return (X.transpose() * X).ldlt().solve(X.transpose() * Y);
}


void parseKernelConfigAttribute(std::string directoryName, std::unordered_map<std::string, KernelConfigAttribute> &kernel_config, void (*parseConfig)(std::string, std::unordered_map<std::string, KernelConfigAttribute>&)){
	std::filesystem::path directory_path {directoryName}; 
	if (!std::filesystem::exists(directory_path) || !std::filesystem::is_directory(directory_path)) {
        	std::cerr << "Error: Directory not found or not a valid directory." << std::endl;
        	return;
    	}
	for (const auto& dirEntry : recursive_directory_iterator(directory_path)){
		if (std::filesystem::is_regular_file(dirEntry.status())){
			parseConfig(static_cast<std::string>(dirEntry.path()), kernel_config);
		}
	}
	
}



void parseInput(std::string fileName, std::unordered_map<std::string, KernelConfigAttribute> &kernel_config){
    	std::ifstream file(fileName);
	if(!file.is_open()) std::cerr << "File " << fileName << " could not be opened.\n" << std::endl;
	
	json data;
	file >> data;
	std::filesystem::path p(fileName);
	fileName = p.stem();
	auto pos {fileName.find('_')};
	std::string modelName {fileName.substr(0, pos)};
	
	for(const auto &[key_name, value]: data[modelName].items()){
		KernelConfigAttribute new_attr;
		for(const auto &item: value["config"].items()){
			if(fieldSetters.find(item.key()) == fieldSetters.end()) std::cout << "item.key() " << item.key() << "\n";
			else fieldSetters[item.key()](new_attr, item.value());
		}
		kernel_config.insert({key_name, new_attr});
	}
	
}

void parsePower(std::string fileName, std::unordered_map<std::string, KernelConfigAttribute> &kernel_config){
	std::ifstream file(fileName);
	if(!file.is_open()) std::cerr << "File " << fileName << " could not be opened.\n" << std::endl;
  	json data = json::parse(file);
  	file.close();
	
	std::filesystem::path p(fileName);
	fileName = p.stem();
	auto pos {fileName.find('_')};
	std::string modelName {fileName.substr(0, pos)};
	for(const auto &[key_name, value]: data[modelName].items()){
		if(!kernel_config.count(key_name)) std::cout << "kernel config doesn't have access! to " << key_name << "\n";
		else{
			std::string strVal = value["power"].get<std::string>();
			double val = std::stod(strVal);
			kernel_config[key_name].power = val;
		}
	}
	
}

void parseLatency(std::string fileName, std::unordered_map<std::string, KernelConfigAttribute> &kernel_config){
    	std::ifstream file(fileName);
	if(!file.is_open()) std::cerr << "File " << fileName << " could not be opened.\n" << std::endl;
  	json data = json::parse(file);
  	file.close();
	std::filesystem::path p(fileName);
	fileName = p.stem();
	auto pos {fileName.find('_')};
	std::string modelName {fileName.substr(0, pos)};

	for(const auto &[key_name, value]: data[modelName].items()){
		if(!kernel_config.count(key_name)) std::cout << "kernel config doesn't have access! to " << key_name << "\n";
		else{
			std::string strVal = value["latency"].get<std::string>();
			double val = std::stod(strVal);
			kernel_config[key_name].latency = val;
		}
	}
	
}


void parseEnergy(std::string fileName, std::unordered_map<std::string, KernelConfigAttribute> &kernel_config){
    	std::ifstream file(fileName);
	if(!file.is_open()) std::cerr << "File " << fileName << " could not be opened.\n" << std::endl;
  	json data = json::parse(file);
  	file.close();
	
	std::filesystem::path p(fileName);
	fileName = p.stem();
	auto pos {fileName.find('_')};
	std::string modelName {fileName.substr(0, pos)};

	for(const auto &[key_name, value]: data[modelName].items()){
		if(!kernel_config.count(key_name)) std::cout << "kernel config doesn't have access! to " << key_name << "\n";
		else{
			std::string strVal = value["energy"].get<std::string>();
			double val = std::stod(strVal);
			kernel_config[key_name].energy = val;
		}
	}
}

void printKernelConfigAttributes(std::unordered_map<std::string, KernelConfigAttribute> &kernel_config){
	for(const auto &pair: kernel_config){

     		std::cout << pair.first << "\t" << pair.second.HW << "\t" << pair.second.CIN << "\t" << pair.second.CIN1; 
		std::cout  << pair.second.CIN2 << "\t" << pair.second.CIN3 << "\t" << pair.second.CIN4; 
		std::cout  << pair.second.COUT << "\t" << pair.second.KERNEL_SIZE << "\t" << pair.second.STRIDES; 
		std::cout  << pair.second.POOL_STRIDES << "\t" << pair.second.power << "\t" << pair.second.latency << "\t" << pair.second.energy << "\n";
	}
}


void printKernelPower(std::unordered_map<std::string, KernelConfigAttribute> &kernel_config){
	std::cout << "printKernelConfigAttributes called\n";
	for(const auto &pair: kernel_config){
     		std::cout << pair.second.HW << "\t" << pair.second.CIN << "\t" << pair.second.power << "\n"; 
	}
}


void printKernelEnergy(std::unordered_map<std::string, KernelConfigAttribute> &kernel_config){
	std::cout << "printKernelConfigAttributes called\n";
	for(const auto &pair: kernel_config){
     		std::cout << pair.second.HW << "\t" << pair.second.CIN << "\t" << pair.second.energy << "\n"; 
	}
}

void printKernelLatency(std::unordered_map<std::string, KernelConfigAttribute> &kernel_config){
	std::cout << "printKernelConfigAttributes called\n";
	for(const auto &pair: kernel_config){
     		std::cout << pair.second.HW << "\t" << pair.second.CIN << "\t" << pair.second.latency << "\n"; 
	}
}
