#include <iostream>
#include <Eigen/Dense>
#include <random>

class Layer {
public:
	Eigen::MatrixXd weights;
	Eigen::VectorXd bias;

	Eigen::VectorXd last_input;
	Eigen::VectorXd last_output;

	Layer(int in_size, int out_size, int seed) {
		std::mt19937 gen(seed);
		std::normal_distribution<double> dist(0.0, 0.5);

		weights.resize(in_size, out_size);
		bias = Eigen::VectorXd::Zero(out_size);
		for (int i = 0; i < weights.rows(); ++i) {
			for (int j = 0; j < weights.cols(); ++j) {
				weights(i, j) = dist(gen);
			}
		}
	}

	double sigmoid(double z) {
		return 1.0 / (1.0 + std::exp(-z));
	}

	Eigen::VectorXd forward(const Eigen::VectorXd input) {
		last_input = input;

		Eigen::VectorXd z = (weights.transpose() * input) + bias;

		last_output = z.unaryExpr([this](double v) { return sigmoid(v); });

		return last_output;
	}
};

class MLP {
	std::vector<Layer> Layers;
public:
	void add_layer(int in_size, int out_size) {
		Layers.emplace_back(in_size, out_size, 123 + Layers.size());
	}
	Eigen::VectorXd predict(Eigen::VectorXd X) {
		for (int i = 0; i < Layers.size(); ++i) {
			X = Layers[i].forward(X);
		}
		return X;
	}

	void train(const Eigen::VectorXd& input, const Eigen::VectorXd& target, double eta) {
		Eigen::VectorXd output = predict(input);
		Eigen::VectorXd delta = (target - output).array() * output.array() * (1.0 - output.array());

		for (int i = Layers.size() - 1; i >= 0; --i) {
			Layer& L = Layers[i];
			Eigen::MatrixXd weight_update = eta * (L.last_input * delta.transpose());
			L.weights += weight_update;
			L.bias += eta * delta;
			if (i > 0) {
				Eigen::VectorXd prev_output = Layers[i - 1].last_output;
				delta = (L.weights * delta).array() * prev_output.array() * (1.0 - prev_output.array());
			}
		}
	}
};

int main() {
	MLP net;
	net.add_layer(2, 4);
	net.add_layer(4, 1);

	for (int i = 0; i < 1000; ++i) {
		std::mt19937 gen(2);
		std::normal_distribution<double> dist(-1.0, 1.0);

		double x = dist(gen);
		double y = dist(gen);

		Eigen::VectorXd input(2);
		input << x, y;

		Eigen::VectorXd target(1);
		if (x * x + y * y <= 0.5) {
			target << 1.0;
		}
		else {
			target << 0.0;
		}

		net.train(input, target, 0.1);
	}

	double user_x, user_y;
	std::cout << "Podaj x: ";
	std::cin >> user_x;
	std::cout << "Podaj y: ";
	std::cin >> user_y;

	Eigen::VectorXd input_point(2);
	input_point << user_x, user_y;

	Eigen::VectorXd result = net.predict(input_point);

	double probability = result(0);

	if (probability > 0.5) {
		std::cout << "PUNKT JEST W KOLE(Prawdopodobienstwo: " << probability * 100 << "%)" << std::endl;
	}
	else {
		std::cout << "PUNKT JEST POZA KOLEM (Prawdopodobienstwo: " << (1.0 - probability) * 100 << "%)" << std::endl;
	}

	return 0;
}