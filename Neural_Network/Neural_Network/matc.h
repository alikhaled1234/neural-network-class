#pragma once
#include<iostream>
#include<random>
#include<ctime>
template<typename T>



class matc
{
private: 

	enum activation {
		None = 0,
		SIGMOID = 1,
		RELU = 2,
		LEAKYRELU = 3,
		SOFTMAX = 4
	};

	T* matrix;
	int row, col;
	int getIndex(int row, int col) const;
	static T relu(T n);
	static T sigmoid(T n);
	static T LeakyRelu(T n, double alpha = 0.1);
	static void softMax(const matc<T>& mat, matc<T>& mat2);
	double xavierInit(int in_size, int out_size);

public:

	T getElement(int row, int col) const;
	void resize(int row, int col, int random, int in = 1, int out = 1);
	void setElement(int row, int col, T element);
	void concat(matc<T>, bool col);
	void transpose();
	std::pair<int, int>getSize() const;
	matc();
	matc(int rows, int cols);
	matc(const matc<T>& matrix);
	matc(const std::vector<std::vector<T>>& v);
	matc<T> dot(const matc<T>& other, int activation, double leakyRealuAlpha = 0.1);
	matc<T> operator*(const matc<T>& other);
	matc<T>& operator=(const matc<T>& mat);
	//matc<T>& operator=(const std::vector<std::vector<T>> &v);
	static void activate(const matc<T>& mat, matc<T>& mat2, int a);
	//matc<T> sum(const matc<T>& other);
	//static matc randoms(int rows, int cols);
	void print() const;
	double rands();
	matc<T> getRow(int row) const;
	~matc();

};

