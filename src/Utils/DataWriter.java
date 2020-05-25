package Utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;

import NeuronalNetwork.Netz;
import NeuronalNetwork.Neuron;

public class DataWriter {

	public static boolean writeNetToFile(String filename, Netz neuralNetwork) {
		
		try {
			
			StringBuilder outputString = new StringBuilder();
			
			// get current date for fileheader
		    Date date = Calendar.getInstance().getTime();  
		    DateFormat dateFormat = new SimpleDateFormat("dd.MM.yyyy");  
		    String strDate = dateFormat.format(date);  
			
			// write a file-header
			outputString.append("+-----------------------------------------------------------+\n");
			outputString.append("|  this file contains a neural network. do not alter data.  |\n");
			outputString.append("|                last modified on " + strDate + "                |\n");
			outputString.append("+-----------------------------------------------------------+\n");
			outputString.append("\n");
			
			// write a explanation for the file structure
			outputString.append("each line contains all weights from one neuron separated by comma.\n"
					+ "the last value represents the bias for that neuron.\n"
					+ "there will be an empty line after all hidden neurons, before the output\n"
					+ "neurons start.\n\n");
			
			// write how many neurons are used
			int[] numberOfNeurons = neuralNetwork.getNumberOfNeurons();
			outputString.append("learnrate = " + neuralNetwork.getLearnrate() + "\n");
			outputString.append("# input neurons = " + numberOfNeurons[0] + "\n");
			outputString.append("# hidden neurons = " + numberOfNeurons[1] + "\n");
			outputString.append("# output neurons = " + numberOfNeurons[2] + "\n");
			outputString.append("\n");
			
			// write the hidden neurons
			ArrayList<Neuron> hiddenNeurons = neuralNetwork.getHiddenNeurons();
			double[] weights;
			for(int i = 0; i < hiddenNeurons.size(); i++) {
				
				// append weights
				weights = hiddenNeurons.get(i).getWeights();
				for(int j = 0; j < weights.length; j++) {
					outputString.append(weights[j] + ",");
				}
				
				// append bias
				outputString.append(hiddenNeurons.get(i).getBias() + "\n");
			}
			
			outputString.append("\n");
			
			// write the output Neurons
			ArrayList<Neuron> outputNeurons = neuralNetwork.getOutputNeurons();
			for(int i = 0; i < outputNeurons.size(); i++) {
				
				// append weights
				weights = outputNeurons.get(i).getWeights();
				for(int j = 0; j < weights.length; j++) {
					outputString.append(weights[j] + ",");
				}
				
				// append bias
				outputString.append(outputNeurons.get(i).getBias() + "\n");
			}
			
//			System.out.println(outputString.toString());
			writeToFile(filename, outputString.toString(), false);

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}
		return true;
	}
	
	private static void writeToFile(String filename, String output, boolean append) throws IOException {

		BufferedWriter writer = new BufferedWriter(new FileWriter(filename, append));
		writer.write(output);
		writer.close();
		
		System.out.println("successfully written to file " + filename);

	}

}
