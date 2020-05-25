package NeuronalNetwork;

import java.util.ArrayList;

public class Neuron {

	double[] weight;
	double bias;
	// TODO: delete? whats its use? maybe for backpropagation?
//	ArrayList<Neuron> nextLayer;

	public Neuron(ArrayList<Neuron> nextLayer, int numberOfWeights) {
//		this.nextLayer = nextLayer;
		weight = new double[numberOfWeights];
		
		// randomize weights
		for(int i = 0; i < numberOfWeights; i++) {
			weight[i] = Math.random();
		}
		// randomize bias
		bias = Math.random();
	}
	
	public Neuron(ArrayList<Neuron> nextLayer, double[] weights, double bias) {
//		this.nextLayer = nextLayer;
		weight = weights;
		this.bias = bias;
	}

	public double fire(double[] input) {

//		System.out.println("input.length = " + input.length + " | weight.length = " + weight.length);
		assert input.length == weight.length;
		
		double sum = 0;
		double length = input.length;

		for (int i = 0; i < length; i++) {
			sum += weight[i] * input[i];
		}
		sum += bias;

		sum = 1 / (1 + Math.exp(-sum));
		
		return sum;
	}
	
	/*
	 * GETTERS
	 */
	public double[] getWeights() {
		return weight;
	}
	
	public double getBias() {
		return bias;
	}
}
