#include "neuralNetwork.h"

neuralNetwork::neuralNetwork(int layerNum, int layerNodes[]) {
	activatedNodes = new matc<double>[layerNum];
	notActivatedNodes = new matc<double>[layerNum];
	dC_dZ = new matc<double>[layerNum];
	dC_dW = new matc<double>[layerNum];
	weights = new matc<double>[layerNum];
	this->numOfLayers = layerNum;
	this->numofLayerNodes = new short[layerNum];
	this->activationFunc = new short[layerNum];
	costFunction = MSE;
	for (int i = 0; i < layerNum; i++) {
		
		if(i != 0) {
			if (i != layerNum - 1) {
				weights[i].resize(layerNodes[i - 1] + 1, layerNodes[i], 2, numofLayerNodes[i - 1], numofLayerNodes[i + 1]);
			}
			else {
				weights[i].resize(layerNodes[i - 1] + 1, layerNodes[i], 2, numofLayerNodes[i - 1], numofLayerNodes[i]);

			}
			
			dC_dW[i].resize(layerNodes[i - 1] + 1, layerNodes[i], 0);
		}
		if (i == layerNum - 1) {
			activatedNodes[i].resize(1, layerNodes[i], 0);
			activationFunc[i] = NONE;
		}
		else {
			activatedNodes[i].resize(1, layerNodes[i], 0);
			activatedNodes[i].concat(matc<double>(), true);
			activationFunc[i] = SIGMOID;
		}
		
		
		notActivatedNodes[i].resize(1, layerNodes[i], 1);
		dC_dZ[i].resize(1, layerNodes[i] + 1, 0);

		this->numofLayerNodes[i] = layerNodes[i];
	}
}

double neuralNetwork::dActivate(double num, short function)
{
	if (function == SIGMOID) {
		return dSigmoid(num);
	}
	else if (function == RELU) {
		return dRELU(num);
	}
	else if (function == LEAKYRELU) {
		return dLeakyRELU(num);
	}
	else if (function == SOFTMAX && costFunction == CROSSENTROPY) {
		return 1.0;
	}
	else if (function == SOFTMAX) {
		return dSoftMax(num);
	}
	else {
		return 1.0;
	}

}

matc<double> neuralNetwork::forwardPass()
{
	for (int i = 0; i < numOfLayers - 1; i++) {

		notActivatedNodes[i + 1] = activatedNodes[i] * weights[i + 1];
		//notActivatedNodes[i + 1].print();
		if (i + 1 != numOfLayers - 1) {
			matc<double>::activate(notActivatedNodes[i + 1], activatedNodes[i + 1], activationFunc[i + 1]);
		}
		else {
			matc<double>::activate(notActivatedNodes[i + 1], activatedNodes[i + 1], activationFunc[i + 1]);
			//std::cout << "--------------------------------------------------\n";
			//std::cout << activationFunc[i + 1] << '\n';
			//notActivatedNodes[i + 1].print();
		}
	}


	return activatedNodes[numOfLayers - 1];
}

double neuralNetwork::dRELU(double a)
{
	return a > 0? 1 : 0;
	
}

double neuralNetwork::dLeakyRELU(double a)
{
	return a != leakyReluAlpha ? 1 : leakyReluAlpha;
}

double neuralNetwork::dSigmoid(double sigmoid)
{
	return sigmoid * (1 - sigmoid);
}

double neuralNetwork::dSoftMax(double soft)
{
	return soft * (1 - soft);
}

double neuralNetwork::dcost(double target, double prediction)
{
	if (costFunction == MSE) {
		return (prediction - target);
	}
	else if (costFunction == CROSSENTROPY && activationFunc[numOfLayers - 1] == SOFTMAX) {
		return prediction - target;
	}
}

void neuralNetwork::updateWeights(double learningRate, int dataSize)
{
	double val;
	for (int i = 1; i < numOfLayers; i++) {
		dC_dW[i] = dC_dW[i] * (1.0 / (double)dataSize);
		for (int j = 0; j < numofLayerNodes[i - 1] + 1; j++) {
			for (int k = 0; k < numofLayerNodes[i]; k++) {
				//std::cout << "weights : L: " << i << " T: " << k << " F: " << j << " " << weights[i].getElement(j, k) << " ";
				if (momentum) {
					val = beta * moment[i].getElement(j, k) + (1.0 - beta) * dC_dW[i].getElement(j, k);
					weights[i].setElement(j, k, weights[i].getElement(j, k) - (val * learningRate));
					moment[i].setElement(j, k, val);
					//std::cout << beta;
				}
				else {

					weights[i].setElement(j, k, weights[i].getElement(j, k) - (dC_dW[i].getElement(j, k) * learningRate));
				}
				//std::cout << weights[i].getElement(j, k) << " " << (dC_dW[i].getElement(j, k) * learningRate) << "\n";
			}
		}

		dC_dW[i] = dC_dW[i] * 0.0;
	}

}

//-------------------------------------------------
template<typename T>
void neuralNetwork::backPropagation(const matc<T>& target)
{
	unsigned short layer = numOfLayers - 1;
	double ans = 1.0;
	for (int j = 0; j < numofLayerNodes[layer]; j++) {
		for (int i = 0; i < numofLayerNodes[layer - 1] + 1; i++) {

			ans = activatedNodes[layer - 1].getElement(0, i);
			if (i == 0) {
				dC_dZ[layer].setElement(0, j, dActivate(activatedNodes[layer].getElement(0, j), activationFunc[layer]) * dcost(target.getElement(0, j), activatedNodes[layer].getElement(0, j)));
			}
			ans *= dC_dZ[layer].getElement(0, j);
			dC_dW[layer].setElement(i, j, ans + dC_dW[layer].getElement(i, j));
			//std::cout << "der : F: " << i << " T: " << j << ", " << dC_dW[layer].getElement(i, j) << '\n';
			

		}
	}
	for(int i = layer - 1; i > 0; i--) {
		for (int k = 0; k < numofLayerNodes[i]; k++) {
			for (int j = 0; j < numofLayerNodes[i - 1] + 1; j++) {
				//std::cout << "hnaqaaa";
				if (j == 0) {
					ans = dActivate(activatedNodes[i].getElement(0, k), activationFunc[i]);
					double tmp = 0.0;
					for (int l = 0; l < numofLayerNodes[i + 1]; l++) {
						tmp += weights[i + 1].getElement(k, l) * dC_dZ[i + 1].getElement(0, l);
					}
					ans *= tmp;
				
					dC_dZ[i].setElement(0, k, ans);
				}
				else {
					ans = dC_dZ[i].getElement(0, k);
				}
				
					//std::cout << ans << " rdsfwe\n ";
				ans *= activatedNodes[i - 1].getElement(0, j);
				dC_dW[i].setElement(j, k, ans + dC_dW[i].getElement(j, k));

			}
		}
	}

}

//template<typename T>
void neuralNetwork::train(const matc<double>& X, const matc<int>& Y, int iterations, double learningRate, int batchSize)
{
	int batchNum = X.getSize().first / batchSize;
	
	if (X.getSize().first % batchSize != 0) {
		batchNum++;
	}
	
	for (int iter = 0; iter < iterations; iter++) {
		this->error = 0.0;
		double accuracy = 0.0;
		for (int k = 0; k < batchNum; k++) {
			for (int i = k * batchSize; i < std::min(X.getSize().first, (k + 1) * batchSize); i++) {
				for (int j = 0; j < X.getSize().second; j++) {

					activatedNodes[0].setElement(0, j, X.getElement(i, j));

				}
				activatedNodes[0].setElement(0, X.getSize().second, 1);
				forwardPass();
				calcError(Y.getRow(i));
				if (Y.getElement(i, getPred()) == 1) {
					accuracy++;
				}
				//backPropagation(Y.getElement(i, 0));
				backPropagation(Y.getRow(i));
			}
			updateWeights(learningRate, Y.getSize().first);
		}
		accuracy /= (double)X.getSize().first;
		std::cout << "Accuracy : " << accuracy * 100.0 << '\n';
		//std::cout << " Error : " << error / X.getSize().first << '\n';
	}
}


matc<double> neuralNetwork::predict(matc<double>&X) {
	matc<double>prediction(X.getSize().first, 1);
	for (int i = 0; i < X.getSize().first; i++) {

		for (int j = 0; j < activatedNodes[0].getSize().second - 1; j++) {
			activatedNodes[0].setElement(0, j, X.getElement(i, j));
		}

		activatedNodes[0].setElement(0, activatedNodes[0].getSize().second - 1, 1);
		forwardPass();

		/*for (int j = 0; j < numofLayerNodes[numOfLayers - 1]; j++) {
			prediction.setElement(i, j, getPred());
		}*/
		prediction.setElement(i, 0, getPred());
	}
	return prediction;
}

double neuralNetwork::getPred() {
	if (activationFunc[numOfLayers - 1] == SOFTMAX) {
		double maxx = activatedNodes[numOfLayers - 1].getElement(0, 0);
		int index = 0;
		for (int i = 1; i < numofLayerNodes[numOfLayers - 1]; i++) {
			//maxx = std::max(maxx, activatedNodes[numOfLayers - 1].getElement(0, i));
			if (activatedNodes[numOfLayers - 1].getElement(0, i) > maxx) {
				maxx = activatedNodes[numOfLayers - 1].getElement(0, i);
				index = i;
			}
		}
		return index;
	}
	else if (activationFunc[numOfLayers - 1] == MSE) {
		return activatedNodes[numOfLayers - 1].getElement(0, 0);
	}
}

double neuralNetwork::getError()
{
	return error;
}

void neuralNetwork::setActivations(std::string activations[]) {
	for (int i = 0; i < numOfLayers; i++) {
		if (activations[i] == "relu") {
			activationFunc[i] = RELU;
		}
		else if (activations[i] == "sigmoid") {
			activationFunc[i] = SIGMOID;
		}
		else if (activations[i] == "leakyRelu") {
			activationFunc[i] = LEAKYRELU;
		}
		else if (activations[i] == "softMax") {
			activationFunc[i] = SOFTMAX;
		}
		else {
			activationFunc[i] = NONE;
		}
	}
}

void neuralNetwork::activateMomentum(double beta) {
	this->beta = beta;
	momentum = true;
	moment = new matc<double>[numOfLayers];
	for (int i = 1; i < numOfLayers; i++) {
		moment[i].resize(numofLayerNodes[i - 1] + 1, numofLayerNodes[i], 0);
	}
}

void neuralNetwork::setCostFunction(std::string c) {
	if (c == "crossEntropy") {
		costFunction = CROSSENTROPY;
	}
	else if (c == "MSE") {
		costFunction = MSE;
	}
}

template<typename T>
void neuralNetwork::calcError(const matc<T>& realVal)
{
	if (costFunction == MSE) {
		for (int i = 0; i < numofLayerNodes[numOfLayers - 1]; i++) {
			error += calcMSE(realVal.getElement(0, i), i);
		}
		
	}
	else if (costFunction == CROSSENTROPY) {
		for (int i = 0; i < numofLayerNodes[numOfLayers - 1]; i++) {
			error += calcCrossEntropy(realVal.getElement(0, i), i);
		}
	}
}

double neuralNetwork::calcCrossEntropy(double Y, int node) {
	return -Y * log(activatedNodes[numOfLayers - 1].getElement(0, node));
}

double neuralNetwork::calcMSE(double Y, int node) {
	return pow(Y - activatedNodes[numOfLayers - 1].getElement(0, node), 2);
}

neuralNetwork::~neuralNetwork() {
	// Deallocate dynamically allocated arrays
	delete[] activatedNodes;
	delete[] notActivatedNodes;
	delete[] dC_dZ;
	delete[] dC_dW;
	delete[] weights;
	delete[] numofLayerNodes;
	delete[] activationFunc;
	if (moment != NULL) {
		delete[] moment;
	}
}


