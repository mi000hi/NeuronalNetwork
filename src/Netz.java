import java.util.ArrayList;

public class Netz {

	ArrayList<Neuron> input = new ArrayList<>();
	ArrayList<Neuron> hidden = new ArrayList<>();
	ArrayList<Neuron> output = new ArrayList<>();
	
	public Netz(int input, int hidden, int output) {
		
		for(int i = 0; i < input; i++) {
			this.input.add(new Neuron(this.hidden));
		}
		
		for(int i = 0; i < hidden; i++) {
			this.hidden.add(new Neuron(this.output));
		}
		
		for(int i = 0; i < output; i++) {
			this.output.add(new Neuron(null));
		}
	}
	
	public double[] compute(double[] inputVector) {
		
		double[] inputResult = new double[this.input.size()];
		double[] hiddenResult = new double[this.hidden.size()];
		double[] outputResult = new double[this.output.size()];
		
		for(int i = 0; i < this.input.size(); i++) {
			inputResult[i] = this.input.get(i).fire(inputVector);
		}
		
		for(int i = 0; i < this.hidden.size(); i++) {
			hiddenResult[i] = this.hidden.get(i).fire(inputResult);
		}
		
		for(int i = 0; i < this.output.size(); i++) {
			outputResult[i] = this.output.get(i).fire(hiddenResult);
		}
		
		return outputResult;
	}
}
