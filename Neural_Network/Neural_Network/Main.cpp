#pragma region Improvements
/*
1.vectorizing what can be
2.using templates in the neural network class
3.making class for the evaluating metrics


*/
#pragma endregion

#include<iostream>
#include<SFML/Graphics.hpp>
#include<vector>
#include<math.h>
#include<fstream>
#include <sstream>
#include"matc.cpp"
#include"neuralNetwork.h"

#define WHITE Color(255, 255, 255)
#define BLACK Color(0, 0, 0)
#define PI 3.14

using namespace sf;
using namespace std;
float width = 500, height = 500;

double toRad(double angle) {
	return angle * PI / 180.0;
}
//std::ostream& operator<<(std::ostream& os, cvec& obj) {
//	int i = 0, j = 0;
//	if (!obj.T) {
//		NESTED_FOR(i, j, obj.size().first, obj.size().second) {
//			os << obj(i, j) << ' ';
//			if (j == obj.size().second - 1) {
//				std::cout << std::endl;
//			}
//		}
//	}
//	else {
//		NESTED_FOR_T(i, j, obj.size().first, obj.size().second) {
//			os << obj(i, j) << ' ';
//			if (i == obj.size().first - 1) {
//				std::cout << std::endl;
//			}
//		}
//	}
//
//	return os;
//}


std::vector<std::vector<int>> readCSV(const std::string& filename) {
	std::vector<std::vector<int>> data;
	std::ifstream file(filename);

	if (!file.is_open()) {
		std::cerr << "Error: Could not open file " << filename << std::endl;
		return data;
	}

	std::string line;
	while (std::getline(file, line)) {
		std::stringstream lineStream(line);
		std::string cell;
		std::vector<int> row;
		while (std::getline(lineStream, cell, ',')) {
			row.push_back(stoi(cell));
		}
		data.push_back(row);
	}

	file.close();
	return data;
}

int twoToOne(int row, int col) {
	return 4 * (col + (width * row));
}

int main() {
	RenderWindow w(VideoMode(width, height), "ali");
	Event e;
	
	int a[] = { 784, 16, 10 };
	sf::Uint8* pixels = new sf::Uint8[width * height * 4];


	neuralNetwork nn(3, a);
	string activs[] = { "relu", "relu", "softMax"};
	nn.setActivations(activs);
	nn.setCostFunction("crossEntropy");
	nn.activateMomentum();

	std::ifstream file("fashion-mnist_train.csv");

	if (!file.is_open()) {
		std::cerr << "Error: Could not open file " << "filename" << std::endl;
		//return data;
	}

	std::string line;
	bool firstLine = true, label = true; 
	vector<vector<double>> csv;
	vector<vector<int>> labels;

	while (std::getline(file, line)) {
		std::stringstream lineStream(line);
		std::string cell;
		std::vector<double> row;
		std::vector<int> col;
		label = true;
		if (!firstLine) {
			while (std::getline(lineStream, cell, ',')) {
				if (label) {
					label = false;
					col.push_back(stoi(cell));
				}
				else {
					row.push_back(stod(cell));
				}
			}
			csv.push_back(row);
			labels.push_back(col);
		}
		else {
			firstLine = false;
		}
	}

	file.close();

	matc<double> train_X(csv);
	vector<vector<int>> l(labels.size(), vector<int>(10, 0));
	for (int i = 0; i < labels.size(); i++) {
		l[i][labels[i][0]] = 1;
	}
	matc<int> train_Y(l);
	train_X = train_X * double(1.0 / 255.0);
	
	cout << " training will start----------------------------\n ";
	int i = 0;
	bool train = true;


	//testing dataa
#pragma region testingData



	firstLine = true, label = true;
	csv.clear();
	labels.clear();
	file.open("fashion-mnist_test.csv");

	if (!file.is_open()) {
		std::cerr << "Error: Could not open file " << "filename" << std::endl;
		//return data;
	}

	
	while (std::getline(file, line)) {
		std::stringstream lineStream(line);
		std::string cell;
		std::vector<double> row;
		std::vector<int> col;
		label = true;
		if (!firstLine) {
			while (std::getline(lineStream, cell, ',')) {
				if (label) {
					label = false;
					col.push_back(stoi(cell));
				}
				else {
					row.push_back(stod(cell));
				}
			}
			csv.push_back(row);
			labels.push_back(col);
		}
		else {
			firstLine = false;
		}
	}

	file.close();
	matc<double> test_X(csv);
	/*vector<vector<int>> l2(labels.size(), vector<int>(10, 0));
	for (int i = 0; i < labels.size(); i++) {
		l[i][labels[i][0]] = 1;
	}*/
	matc<int> test_Y(labels);
	test_X = test_X * double(1.0 / 255.0);
#pragma endregion
	sf::Texture t;
	sf::Sprite s;

	t.create(width, height);
	
	while (w.isOpen()) {
		//nn.train(train_X, train_Y, 1, 0.1, train_X.getSize().first);
		if (train) {
			nn.train(train_X, train_Y, 1, 1, 100);
			cout << nn.getError() << '\n';
		}
		else {
			int n;
			cin >> n;
			if (n == -2) {
				matc<double>abc = nn.predict(test_X);
				double acc = 0;
				for (int i = 0; i < abc.getSize().first; i++) {
					if (abc.getElement(i, 0) == test_Y.getElement(i, 0)) {
						acc++;
					}
				}
				acc /= abc.getSize().first;
				cout << "acc : " << acc << '\n';
			}
			else if (n < 10000 && n > 0) {
				matc<double> a = test_X.getRow(n);
				a = nn.predict(a);
				cout << "pred : " << a.getElement(0, 0) << " actual: " << labels[n][0] << "\n";
				memset(pixels, 0, width * height * 4 * sizeof(sf::Uint8));
				/*int ind = 0;
				for (int i = 0; i < 784 * 4; i+=4) {
					cout << ind << " ";
					pixels[i + 100000] = csv[n][ind];
					pixels[i+1 + 100000] = csv[n][ind];
					pixels[i+2 + 100000] = csv[n][ind++];
					pixels[i+3 + 100000] = 255;
				}*/
				for (int i = 0; i < 28; i++) {
					for (int j = 0; j < 28; j++) {
						pixels[twoToOne(225 + i, 225 + j)] = csv[n][j + (28 * i)];
						pixels[twoToOne(225 + i, 225 + j) + 1] = csv[n][j + (28 * i)];
						pixels[twoToOne(225 + i, 225 + j) + 2] = csv[n][j + (28 * i)];
						pixels[twoToOne(225 + i, 225 + j) + 3] = 255;
					}
				}
				t.update(pixels);
				s.setTexture(t);
			}
			else if (n == -1) {
				train = true;
			}
		}
		while (w.pollEvent(e)) {
			if (e.type == Event::Closed) {
				w.close();
			}
		}
		
			//cout << "hnaaa";
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
		{
			train = false;
		}
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::R))
		{
			train = false;
		}
		/*if (e.type == sf::Event::KeyPressed) {
			if (e.key.code == sf::Keyboard::S) {
				
			}
			else if (e.key.code == sf::Keyboard::R) {
				train = true;
			}
		}*/
		/*t2.create(width, height);
		t2.update(pixels2);
		s2.setTexture(t2);*/
		w.clear(Color::Black);
		w.draw(s);
		/*w.draw(s2);*/
		w.display();
	}
	return 0;
}