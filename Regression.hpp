#pragma once

#include <ostream>

#include <Eigen/Core>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> InputData;
typedef Eigen::Vector<double, Eigen::Dynamic> InputRow;
typedef Eigen::Vector<double, Eigen::Dynamic> OutputData;

class Regression {
	private:
		Eigen::Vector<double, Eigen::Dynamic> weights;
		double bias;
		bool hasBias;
		
	public:
		void train(const InputData& in, const OutputData& out, double learningRate, bool hasBias = true);
		void trainClosedForm(const InputData& in, const OutputData& out, bool hasBias = true);
		double predict(const InputRow& inRow);

	//friends:
		friend std::ostream& operator<<(std::ostream& os, const Regression& reg);
};

std::ostream& operator<<(std::ostream& os, const Regression& reg);
