#include "Regression.hpp"
#include <iostream>

#include <Eigen/Dense>

void Regression::train(const InputData& in, const OutputData& out, double learningRate, bool hasBias = true) {

}

void Regression::trainClosedForm(const InputData& in, const OutputData& out, bool hasBias) {
	this->hasBias = hasBias;

	InputData inCopy = in;

	if(hasBias) {
		inCopy.conservativeResize(inCopy.rows(), inCopy.cols() + 1);
		inCopy.col(inCopy.cols()-1) = Eigen::Vector<double, Eigen::Dynamic>::Ones(inCopy.rows());
	}

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A = inCopy.transpose() * inCopy;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> b = inCopy.transpose() * out;

	weights = A.colPivHouseholderQr().solve(b);

	if(hasBias) {
		bias = weights[weights.size() - 1];
		weights.conservativeResize(weights.size() - 1);
	} else {
		bias = 0;
	}
}

double Regression::predict(const InputRow& inRow) {
	return weights.dot(inRow) + bias; 
}

std::ostream& operator<<(std::ostream& os, const Regression& reg) {
	os << "Weignts: " << reg.weights.transpose() << " | ";
	os << "Bias: " << reg.bias;
	return os;
}
