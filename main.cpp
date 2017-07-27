#include "ffbpneuralnet.h"

#include <iostream>
using std::cout;
using std::endl;

#include <ctime>


int main(void)
{
	// trains a feed-forward back-propagation neural network on how to calculate XOR based on two binary input values
	// ie: 0 XOR 0 == 0, 1 XOR 0 == 1, 0 XOR 1 == 1, 1 XOR 1 == 0

	srand(static_cast<unsigned int>(time(0)));
	//srand(1);

	// create a network of 2 input neurons, one hidden layer of 5 neurons, and 1 output neuron
	vector<size_t> HiddenLayers;
	HiddenLayers.push_back(5);
	FFBPNeuralNet NNet(2, HiddenLayers, 1);


	double max_error_rate = 0.01;
	long unsigned int max_training_sessions = 100000;

	double error_rate = 0.0;
	long unsigned int num_training_sessions = 0;


	// train network until the error rate goes below the maximum error rate
	// or we reach the maximum number of training sessions (which could be considered as "giving up")
	do
	{
		vector<double> inputs;

		inputs.push_back(0.0);
		inputs.push_back(0.0);
		NNet.FeedForward(inputs);
		inputs.clear();
		inputs.push_back(0.0);
		error_rate = NNet.BackPropagate(inputs);

		inputs.clear();
		inputs.push_back(0.0);
		inputs.push_back(1.0);
		NNet.FeedForward(inputs);
		inputs.clear();
		inputs.push_back(1.0);
		error_rate += NNet.BackPropagate(inputs);

		inputs.clear();
		inputs.push_back(1.0);
		inputs.push_back(0.0);
		NNet.FeedForward(inputs);
		inputs.clear();
		inputs.push_back(1.0);
		error_rate += NNet.BackPropagate(inputs);

		inputs.clear();
		inputs.push_back(1.0);
		inputs.push_back(1.0);
		NNet.FeedForward(inputs);
		inputs.clear();
		inputs.push_back(0.0);
		error_rate += NNet.BackPropagate(inputs);

		error_rate /= 4.0;
		num_training_sessions++;
	}
	while(error_rate >= max_error_rate && num_training_sessions < max_training_sessions);


	// print out how many training sessions it took to arrive at whatever the final error rate was
	cout << "Final number of training sessions/epochs: " << num_training_sessions << endl;
	cout << "Final error rate: " << error_rate << endl << endl;


	cout << "Demonstrating newly trained network..." << endl;

	// now use the trained network to obtain results -- hopefully it works (sometimes it does not, so retraining would be necessary)
	vector<double> values;

	values.push_back(0.0);
	values.push_back(0.0);
	NNet.FeedForward(values);
	values.clear();
	NNet.GetOutputValues(values);
	cout << "0 XOR 0 ~ " << values[0] << endl;

	values.clear();
	values.push_back(0.0);
	values.push_back(1.0);
	NNet.FeedForward(values);
	values.clear();
	NNet.GetOutputValues(values);
	cout << "0 XOR 1 ~ " << values[0] << endl;

	values.clear();
	values.push_back(1.0);
	values.push_back(0.0);
	NNet.FeedForward(values);
	values.clear();
	NNet.GetOutputValues(values);
	cout << "1 XOR 0 ~ " << values[0] << endl;

	values.clear();
	values.push_back(1.0);
	values.push_back(1.0);
	NNet.FeedForward(values);
	values.clear();
	NNet.GetOutputValues(values);
	cout << "1 XOR 1 ~ " << values[0] << endl;


	// save the network to a file
	NNet.SaveToFile("network.bin");

	cout << endl;

	// load a network from a file
	FFBPNeuralNet NNet2("network.bin");

	cout << "Demonstrating that the \"save/load network to file\" process is working..." << endl;

	// now use the pre-trained network
	values.clear();

	values.push_back(0.0);
	values.push_back(0.0);
	NNet2.FeedForward(values);
	values.clear();
	NNet2.GetOutputValues(values);
	cout << "0 XOR 0 ~ " << values[0] << endl;

	values.clear();
	values.push_back(0.0);
	values.push_back(1.0);
	NNet2.FeedForward(values);
	values.clear();
	NNet2.GetOutputValues(values);
	cout << "0 XOR 1 ~ " << values[0] << endl;

	values.clear();
	values.push_back(1.0);
	values.push_back(0.0);
	NNet2.FeedForward(values);
	values.clear();
	NNet2.GetOutputValues(values);
	cout << "1 XOR 0 ~ " << values[0] << endl;

	values.clear();
	values.push_back(1.0);
	values.push_back(1.0);
	NNet2.FeedForward(values);
	values.clear();
	NNet2.GetOutputValues(values);
	cout << "1 XOR 1 ~ " << values[0] << endl;

	return 0;
}
