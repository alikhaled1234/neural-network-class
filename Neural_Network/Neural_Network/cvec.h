#pragma once
#include<iostream>
#include<vector>

#define NESTED_FOR(i, j, rows, cols)  \
    for (int i = 0; i < rows; ++i)    \
        for (int j = 0; j < cols; ++j)

#define NESTED_FOR_T(i, j, rows, cols) \
    for (int j = 0; j < cols; ++j)    std::cout << '\n';  \
		for (int i = 0; i < rows; ++i)

class cvec
{
public:
	cvec(float arr[], int size);
	cvec(int rows, int cols, float init);
	cvec(float **arr, int rows, int cols);
	void insert(float arr[], int size, bool row);
	static cvec& dot(cvec v1, cvec v2);
	cvec transpose();
	float& operator()(size_t row, size_t col);
	cvec& operator=(cvec v);
	std::pair<int, int> size();
	bool T = false;

private:
	int row, col;
	int& operator[](size_t index);
	std::vector<std::vector<float>>vec;


};

