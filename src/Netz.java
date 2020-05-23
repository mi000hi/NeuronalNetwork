import java.util.ArrayList;

public class Netz {

	ArrayList<Neuron> inputNeurons = new ArrayList<>();
	ArrayList<Neuron> hiddenNeurons = new ArrayList<>();
	ArrayList<Neuron> outputNeurons = new ArrayList<>();

	double learnrate;

	public Netz(int input, int hidden, int output, double learnrate) {

		this.learnrate = learnrate;

		for (int i = 0; i < input; i++) {
			this.inputNeurons.add(new Neuron(this.hiddenNeurons, 0));
		}

		for (int i = 0; i < hidden; i++) {
			this.hiddenNeurons.add(new Neuron(this.outputNeurons, input));
		}

		for (int i = 0; i < output; i++) {
			this.outputNeurons.add(new Neuron(null, hidden));
		}
	}

	/**
	 * Back-propagation algorithm from:
	 * https://intellipaat.com/blog/tutorial/artificial-intelligence-tutorial/back-propagation-algorithm/
	 * 
	 * bias-correction from:
	 * https://stackoverflow.com/questions/3775032/how-to-update-the-bias-in-neural-network-backpropagation
	 * 
	 * @param input
	 * @param expectedResult
	 * @param repetitions
	 */
	public void train(double[] input, double[] expectedResult, int repetitions) {

		double[] result;
		double[] hidden;
		double dE_dOut_i, dOut_i_net_Out_i, dNet_Out_i_dw_j, dNet_Out_i_dHid_j, dE_total_dHid_j, dHid_j_dNet_Hid_j,
				dNet_Hid_j_dw_k;
		double errorBeforeTraining, error = -1;
		double[] dE_Out_i_dNet_Out_i = new double[outputNeurons.size()];

		System.out.println("Starting training for " + repetitions + " repetitions...");
		result = compute(input);
		errorBeforeTraining = error(expectedResult, result);

//		while(error == -1 || error > 0.0000001) {
		for (int rep = 0; rep < repetitions; rep++) {

			// propagate result for given input
			hidden = computeHidden(input);
			result = computeOutput(hidden);
			error = error(expectedResult, result);

			// get error as reference
//			System.out.println("Error at rep " + rep + ": " + error);

			/*
			 * for each output-neuron, calculate all weights w_j according to
			 * 
			 * d(E_total)/d(w_j) = d(E_total)/d(Out_i) * d(Out_i)/d(net_Out_i) *
			 * d(net_Out_i)/d(w_j) = d(E_Out_i)/d(net_Out_i) * d(net_Out_i)/d(w_j)
			 */
			for (int i = 0; i < outputNeurons.size(); i++) {

				// d(E_total)/d(Out_i)
				dE_dOut_i = -(expectedResult[i] - result[i]);

				// d(Out_i)/d(net_Out_i)
				dOut_i_net_Out_i = result[i] * (1 - result[i]);

				// d(E_Out_i)/d(net_Out_i)
				dE_Out_i_dNet_Out_i[i] = dE_dOut_i * dOut_i_net_Out_i;
			}

			/*
			 * for each hidden-neuron, calculate all weights w_k according to
			 * 
			 * d(E_total)/d(w_k) = d(E_total)/d(Hid_j) * d(Hid_j)/d(Net_Hid_j) *
			 * d(Net_Hid_j)/d(w_k)
			 */
			for (int j = 0; j < hiddenNeurons.size(); j++) {

				// d(E_total)/d(hid_j) = sum_i( d(E_Out_i)/d(hid_j) ) = sum_i(
				// d(E_Out_i)/d(Net_Out_i) * d(Net_Out_i)/d(hid_j) )
				dE_total_dHid_j = 0;
				for (int i = 0; i < outputNeurons.size(); i++) {

					// d(net_Out_i)/d(hid_j)
					dNet_Out_i_dHid_j = outputNeurons.get(i).weight[j];

					dE_total_dHid_j += dE_Out_i_dNet_Out_i[i] * dNet_Out_i_dHid_j;
				}

				// d(Hid_j)/d(Net_Hid_j)
				dHid_j_dNet_Hid_j = hidden[j] * (1 - hidden[j]);

				/** adjust output-weights **/
				for (int k = 0; k < input.length; k++) {

					// d(Net_Hid_j)/d(w_k)
					dNet_Hid_j_dw_k = input[k];

					// w_k = w_k - eta * d(E_total)/d(w_k)
					hiddenNeurons.get(j).weight[k] -= learnrate * dE_total_dHid_j * dHid_j_dNet_Hid_j * dNet_Hid_j_dw_k;
				}
				
				// b_j = b_j - eta * delta(j)
				hiddenNeurons.get(j).bias -= learnrate * dE_total_dHid_j * dHid_j_dNet_Hid_j;
			}

			/** adjust output-weights **/
			for (int i = 0; i < outputNeurons.size(); i++) {
				for (int j = 0; j < hidden.length; j++) {

					// d(net_Out_i)/d(w_i)
					dNet_Out_i_dw_j = hidden[j];

					// w_j = w_j - eta * d(E_total)/d(w_j)
					outputNeurons.get(i).weight[j] -= learnrate * dE_Out_i_dNet_Out_i[i] * dNet_Out_i_dw_j;
				}
				
				// b_i = b_i - eta * delta(i)
				outputNeurons.get(i).bias -= learnrate * dE_Out_i_dNet_Out_i[i];
			}
		}

		// propagate result for given input to get reference error
		result = compute(input);
		error = error(expectedResult, result);

		System.out.println("Error before training: " + errorBeforeTraining);
		System.out.println("Error after training:  " + error);
		
		System.out.println("prediction for " + arrayToString(input) + " is " + arrayToString(result));
	}

	private double error(double[] expectation, double[] reality) {

		assert (expectation.length == reality.length);

		double error = 0;

		for (int i = 0; i < expectation.length; i++) {
			error += 0.5 * Math.pow(expectation[i] - reality[i], 2);
		}

		return error;
	}

	public double[] compute(double[] inputVector) {

		double[] hiddenResult;
		double[] outputResult = new double[this.outputNeurons.size()];

		hiddenResult = computeHidden(inputVector);

		outputResult = computeOutput(hiddenResult);

		return outputResult;
	}

	public double[] computeHidden(double[] inputVector) {

		double[] hiddenResult = new double[this.hiddenNeurons.size()];

		for (int i = 0; i < this.hiddenNeurons.size(); i++) {
			hiddenResult[i] = this.hiddenNeurons.get(i).fire(inputVector);
		}

		return hiddenResult;
	}

	public double[] computeOutput(double[] hiddenOutputVector) {

		double[] outputResult = new double[this.outputNeurons.size()];

		for (int i = 0; i < this.outputNeurons.size(); i++) {
			outputResult[i] = this.outputNeurons.get(i).fire(hiddenOutputVector);
		}

		return outputResult;
	}
	
	private String arrayToString(double[] array) {
		
		String result = "{ ";
		
		for(int i = 0; i < array.length - 1; i++) {
			result += array[i] + ", ";
		}
		result += array[array.length - 1] + " }";
		
		return result;
	}
}
