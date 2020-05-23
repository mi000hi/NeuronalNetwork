import java.util.ArrayList;

public class Neuron {

	double[] gewicht = new double[100];
	double schwellwert;
	double bias;
	ArrayList<Neuron> sendToArrayList;

	public Neuron(ArrayList<Neuron> sendTo) {
		sendToArrayList = sendTo;
	}
	
	public void train(double[] input, int result, int steps) {
		
		double alpha = 0.01;
		
		for(; steps > 0; steps--) {
			int step = fire(input);
			
			if(step == result) {
				return;
			}
			
			for(int i = 0; i < gewicht.length; i++) {
				gewicht[i] = gewicht[i] + alpha * input[i] * (result - step);
			}
		}
	}

	public int fire(double[] input) {

		double sum = 0;
		double length = Math.min(input.length, gewicht.length);

		for (int i = 0; i < length; i++) {
			sum += gewicht[i] * input[i];
		}
		sum += bias;

		if (schwellwert <= sum) {
			return 1;
		}
		return 0;
	}
}
