#include "cvec.h"
cvec::cvec(float arr[], int size) {
	vec.resize(size);
	row = 0, col = size;
	for (int i = 0; i < size; i++) {
		vec[0][i] = arr[i];
	}
}
cvec::cvec(float **arr, int rows, int cols) {
	vec.resize(rows);
	row = rows, col = cols;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			vec[i].push_back(arr[i][j]);
		}
	}
}
cvec::cvec(int rows, int cols, float init) {
	row = rows, col = cols;
	vec.resize(rows);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			vec[i].push_back(init);
		}
	}
}
void cvec::insert(float arr[], int size, bool row) {
	if (row) {

		vec.resize(vec.size() + 1);
		for (int i = 0; i < size; i++) {
			vec[this->row].push_back(arr[i]);
			

		}
		this->row++;
	}
	else {
		for (int i = 0; i < size; i++) {
			vec[i].push_back(arr[i]);
		}
		this->col++;
	}
}

cvec cvec::transpose() {

	T = !T;
	return *(this);
}

cvec& cvec::dot(cvec v1, cvec v2) {
	if (v1.col == v2.row) {
		float num = 0;
		float* arr = new float[v1.col + 1];
		cvec v(0, 0);
		if (!v1.T && !v2.T) {
			for (int i = 0; i < v1.row; i++) {
				for (int j = 0; j < v2.col; j++) {
					for (int k = 0; k < v1.col; k++) {
						num += v1(i, k) * v2(k, j);
					}
					arr[j] = num;
					num = 0;
				}
				v.insert(arr, v2.col, true);
			}
		}
		else if (v1.T && v2.T) {
			for (int i = 0; i < v1.col; i++) {
				for (int j = 0; j < v2.row; j++) {
					for (int k = 0; k < v1.row; k++) {
						num += v1(k, i) * v2(j, k);
					}
					arr[j] = num;
					num = 0;
				}
				v.insert(arr, v2.col, true);
			}
		}
		else if (v2.T) {
			for (int i = 0; i < v1.row; i++) {
				for (int j = 0; j < v2.row; j++) {
					for (int k = 0; k < v1.col; k++) {
						num += v1(i, k) * v2(j, k);
					}
					arr[j] = num;
					num = 0;
				}
				v.insert(arr, v2.col, true);
			}
		}
		else {
			for (int i = 0; i < v1.col; i++) {
				for (int j = 0; j < v2.col; j++) {
					for (int k = 0; k < v1.row; k++) {
						num += v1(k, i) * v2(k, j);
					}
					arr[j] = num;
					num = 0;
				}
				v.insert(arr, v2.col, true);
			}
		}
		
		delete[] arr;
		return v;
	}
}

std::pair<int, int> cvec::size() {
	if (!T) {
		return { row, col };
	}
	else {
		return { col, row };
	}
}



float& cvec::operator()(size_t row, size_t col) {
	if (row >= vec.size() || col >= vec[row].size()) {
		throw std::out_of_range("Index out of range");
	}
	if (this->T) {
		return vec[col][row];
	}
	else {

		return vec[row][col];
	}
}
cvec& cvec::operator=(cvec v) {
	/*if (row >= vec.size() || col >= vec[row].size()) {
		throw std::out_of_range("Index out of range");
	}
	if (this->T) {
		return vec[col][row];
	}
	else {
		return vec[row][col];
	}*/
}
// Overload operator[] for non-const objects (for read and write access)
//int& cvec::operator[](size_t index) {
//	if (index >= this->size) {
//		throw std::out_of_range("Index out of range");
//	}
//	return data[index];
//}