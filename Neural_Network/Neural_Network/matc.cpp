#ifndef MATC_CPP
#define MATC_CPP
#include "matc.h"

template<typename T>
matc<T>::matc(int rows, int cols) {
	srand(time(0));
	int size = cols * rows;
	row = rows, col = cols;
	matrix = new T[size];
	for (int i = 0; i < size; i++) {
		matrix[i] = 0;
	}
}

template<typename T>
matc<T>::matc() {
	srand(time(0));
	row = 1, col = 1;
	matrix = new T[1];
	matrix[0] = 1;
}

template<typename T>
matc<T>::matc(const matc<T>& mat) {
	srand(time(0));
 	row = mat.row, col = mat.col;
	int size = row * col;
	matrix = new T[size];
	for (int i = 0; i < size; i++) {
		this->matrix[i] = mat.matrix[i];
	}

}

template<typename T>
matc<T>::matc(const std::vector<std::vector<T>>& v)
{
	row = v.size();
	col = v[0].size();
	matrix = new T[row * col];
	for (int i = 0; i < v.size(); i++) {
		for (int j = 0; j < v[i].size(); j++) {
			matrix[getIndex(i, j)] = v[i][j];
		}
	}
}

//template <typename T>
//matc<T>::matc(const matc<T>& other) : row(other.row), col(other.col) {
//	matrix = new T[row * col];
//	std::copy(other.matrix, other.matrix + (row * col), matrix);
//}

template<typename T>
int matc<T>::getIndex(int row, int col) const {
	//if ((row * this->col) + col >= this->row * this->col) {
	//	//std::cout << row << " " << col;
	//	throw std::exception("out of bounds");
	//}
	if (row >= this->row || col >= this->col) {
		throw std::exception("out of bounds");
	}
	return (row * this->col) + col;
}

//template<typename T>
//matc<T> matc<T>::randoms(int rows, int cols) {
//	//srand(time(0));
//
//	matc<T> temp(rows, cols);
//	for (int i = 0; i < rows * cols; i++) {
//		temp.matrix[i] = rand() % 7 - 3;
//		
//	}
//	return temp;
//}

template<typename T>
double matc<T>::rands()
{
	//srand(time(0));

	return ((double)(rand() % 2) - 0.5) + ((double)(rand() % 101) / 10000.0);
}

template<typename T>
matc<T> matc<T>::getRow(int row) const
{
	if (row >= this->row) {
		throw std::exception();
	}
	matc<T> tmp(1, col);
	for (int i = 0; i < col; i++) {
		tmp.setElement(0, i, matrix[getIndex(row, i)]);
	}
	return tmp;
}

template<typename T>
double matc<T>::xavierInit(int in_size, int out_size) {
	return sqrt(2.0 / (in_size + out_size)) * ((double)rand() / RAND_MAX - 0.5);
}

template<typename T>
void matc<T>::resize(int rows, int cols, int random, int in, int out) {
	
	delete[] matrix;
	row = rows, col = cols;
	matrix = new T[rows * cols];
	if (random == 1) {
		for (int i = 0; i < rows * cols; i++) {
			matrix[i] = rands();
			
		}

	}
	else if (random == 2) {
		for (int i = 0; i < rows * cols; i++) {
			matrix[i] = xavierInit(in, out);

		}
	}
	else {
		for (int i = 0; i < rows * cols; i++) {
			matrix[i] = 0;
		}
	}
}

template<typename T>
matc<T>& matc<T>::operator=(const matc<T>& mat) {
	if (this == &mat) {
		return *this; // Self-assignment check
	}
	
	delete[] matrix; // Free existing resource
	row = mat.row;
	col = mat.col;
	int size = row * col;
	matrix = new T[size];
	for (int i = 0; i < size; i++) {
		matrix[i] = mat.matrix[i];
	}
	return *this;
}



template<typename T>
void matc<T>::activate(const matc<T>& mat, matc<T>& mat2, int a)
{
	if (a == SIGMOID) {
		for (int i = 0; i < mat.row * mat.col; i++) {
			mat2.matrix[i] = sigmoid(mat.matrix[i]);
		}
	}
	else if (a == RELU) {
		for (int i = 0; i < mat.row * mat.col; i++) {
			mat2.matrix[i] = relu(mat.matrix[i]);
		}
	}
	else if (a == LEAKYRELU) {
		for (int i = 0; i < mat.row * mat.col; i++) {
			mat2.matrix[i] = LeakyRelu(mat.matrix[i]);
		}
	}
	else if (a == SOFTMAX) {
		softMax(mat, mat2);
	}
	else {
		mat2 = mat;
	}
	
}

template<typename T>
void matc<T>::print() const {
	
	for (int i = 0; i < row * col; i++) {
		if (i % col == 0 && i > 0) {
			std::cout << '\n';
		}
			std::cout << matrix[i] << " ";
	}
	std::cout << '\n';
}

template<typename T>
matc<T> matc<T>::operator*(const matc<T>& other) {
	if (this->col != other.row) {
		throw std::exception("incompatible matricies");
	}

	matc<T> temp(this->row, other.col);
	double ans = 0;
	for (int i = 0; i < this->row; i++) {
		for (int j = 0; j < other.col; j++) {
			for (int k = 0; k < this->col; k++) {
				ans += this->matrix[getIndex(i, k)] * other.matrix[other.getIndex(k, j)];
			}
			temp.matrix[temp.getIndex(i, j)] = ans;
			ans = 0;
		}
	}
	
	return temp;
}

template<typename T>
matc<T> matc<T>::dot(const matc<T>& other, int activation, double leakyRealuAlpha)
{
	if (this->col != other.row) {
		throw std::exception("incompatible matricies");
	}

	matc<T> temp(this->row, other.col);
	double ans = 0;
	for (int i = 0; i < this->row; i++) {
		for (int j = 0; j < other.col; j++) {
			for (int k = 0; k < this->col; k++) {
				ans += this->matrix[getIndex(i, k)] * other.matrix[other.getIndex(k, j)];
			}
			if (activation == 1) {
				ans = sigmoid(ans);
			}
			else if (activation == 2) {
				ans = relu(ans);
			}
			else if (activation == 3) {
				ans = LeakyRelu(ans, leakyRealuAlpha);
			}
			temp.matrix[temp.getIndex(i, j)] = ans;
			ans = 0;
		}
	}

	return temp;
}

template<typename T>
matc<T> operator*(matc<T>& lhs, const T& rhs) {


	matc<T> temp(lhs.getSize().first, lhs.getSize().second);
	T ans = 0;
	for (int i = 0; i < lhs.getSize().first; i++) {
		for (int j = 0; j < lhs.getSize().second; j++) {
			temp.setElement(i, j, lhs.getElement(i, j) * rhs);
		}
	}
	return temp;
}

template<typename T>
matc<T> operator*(const T& lhs, matc<T>& rhs) {


	matc<T> temp(rhs.getSize().first, rhs.getSize().second);
	T ans = 0;
	for (int i = 0; i < rhs.getSize().first; i++) {
		for (int j = 0; j < rhs.getSize().second; j++) {
			temp.setElement(i, j, rhs.getElement(i, j) * lhs);
		}
	}
	return temp;
}

template<typename T>
std::pair<int, int> matc<T>::getSize() const
{
	return {row, col};
}

template<typename T>
T matc<T>::getElement(int row, int col) const
{
	return matrix[getIndex(row, col)];
}

template<typename T>
void matc<T>::setElement(int row, int col, T element)
{
	matrix[getIndex(row, col)] = element;
}

template<typename T>
void matc<T>::concat(matc<T> m, bool col)
{
	T* tmp;
	if(!col) {
		tmp = new T[this->col * (row + 1)];
		for (int i = 0; i < row * this->col; i++) {
			tmp[i] = matrix[i];
			//std::cout << tmp[i] << " " << matrix[i] << " ok\n";
		}
		for (int i = row * this->col; i < this->col * (row + 1); i++) {
			tmp[i] = m.matrix[i - (row * this->col)];
		}
		row = row + 1;
		delete[] matrix;
		matrix = tmp;
		tmp = NULL;
	}
	else {
		tmp = new T[row * (this->col + 1)];
		int index = 0, index2 = 0;
		for (int i = 0; i < this->col * row; i++) {
			if (i != 0 && i % this->col == 0) {
				tmp[index++] = m.matrix[index2++];
			}
			tmp[index++] = matrix[i];
		}
		tmp[index] = m.matrix[index2];
		this->col++;
		delete[] matrix;
		matrix = tmp;
		tmp = NULL;
	}
}

template<typename T>
void matc<T>::transpose()
{
	T* tmp = new T[row * col];

	int index = 0;
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++) {
			tmp[index++] = matrix[getIndex(j, i)];
		}
	}
	delete[] matrix;
	matrix = tmp;
	tmp = NULL;
	index = col;
	col = row;
	row = index;
}

template<typename T>
T matc<T>::relu(T n)
{
	if (n <= 0) {
		return 0.0;
	}
	else {
		return n;
	}
}

template<typename T>
T matc<T>::sigmoid(T n)
{
	return 1 / (1 + exp(-n));
}

template<typename T>
T matc<T>::LeakyRelu(T n, double alpha)
{
	if (n <= 0) {
		return alpha;
	}
	else {
		return n;
	}
}

template<typename T>
void matc<T>::softMax(const matc<T>& mat, matc<T>& mat2)
{
	int size = mat.col * mat.row;
	double sum = 0.0;
	double* tmp = new double[size];
	for (int i = 0; i < size; i++) {
		tmp[i] = std::exp(mat.matrix[i]);
		sum += tmp[i];
	}
	double f = 0;
	for (int i = 0; i < size; i++) {
		mat2.matrix[i] = tmp[i] / sum;
		f += mat2.matrix[i];
	}
	/*if (f != 1) {
		for (int i = 0; i < size; i++) {
			std::cout << tmp[i] << " ";
		}
		std::cout << sum << '\n';
	}*/
	delete[]tmp;
}

template<typename T>
matc<T>::~matc() {
	delete[] matrix;
	matrix = nullptr;
}


#endif //MATC_CPP