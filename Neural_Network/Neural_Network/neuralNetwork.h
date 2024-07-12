#pragma once

#include"matc.cpp"
#include<iostream>

class neuralNetwork
{
private:
	//		nodes values / 
	matc<double>* activatedNodes, *notActivatedNodes, *weights, *dC_dZ, *dC_dW, *moment;
	matc<double>inputs;
	short *numofLayerNodes, *activationFunc;
	int numOfLayers = 0;
	double leakyReluAlpha = 0.1, error = 100000.0, beta = 0.9;
	bool momentum = false;
	short costFunction;
	
	enum activation {
		NONE = 0,
		SIGMOID = 1,
		RELU = 2,
		LEAKYRELU = 3,
		SOFTMAX = 4
	};
	enum cost {
		MSE = 1,
		CROSSENTROPY = 2,
		
	};
	double dActivate(double num, short function);
	double dRELU(double a);
	double dLeakyRELU(double a);
	double dSigmoid(double sigmoid);
	double dSoftMax(double soft);
	double dcost(double target, double prediction);
	double dmeanSquared(double target, double prediction);
	double dcrossEntropy(double target, double prediction);
	matc<double> forwardPass();
	template<typename T>
	void backPropagation(const matc<T>& target);
	void updateWeights(double learningRate, int dataSize);
	double softMax();
	template<typename T>
	void calcError(const matc<T>& realVal);
	double calcMSE(double Y, int node);
	double calcCrossEntropy(double Y, int node);
	double getPred();

public:
	neuralNetwork(int layerNum, int layerNodes[]);
	//void getAns();
	//template<typename T>
	void train(const matc<double>& X, const matc<int>& Y, int iterations, double learningRate, int batchSize);
	matc<double>predict(matc<double>&X);
	double getError();
	void setActivations(std::string activations[]);
	void setCostFunction(std::string c);
	void activateMomentum(double moment = 0.9);
	~neuralNetwork();


};

