package Utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;

import NeuronalNetwork.Net;
import NeuronalNetwork.Neuron;

public class DataReader {

	public static Net readNetFromFile(String filename) {
		
		double learnrate = 0;
		int nrInputNeurons = 0, nrHiddenNeurons = 0, nrOutputNeurons = 0;
		double[] weights;
		double bias;
		ArrayList<Neuron> inputNeurons = new ArrayList<>();
		ArrayList<Neuron> hiddenNeurons = new ArrayList<>();
		ArrayList<Neuron> outputNeurons = new ArrayList<>();
		
		String sCurrentLine;
		String[] sWeights;
		boolean readHiddenNeurons = false, readOutputNeurons = false;
		
		InputStream input = null;
		try {
			input = new FileInputStream(new File(filename));
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		// TODO: not sure if this is still necessary, since i used a fileinputstream ^^
//        if (input == null) {
//            // this is how we load file within editor (eg eclipse)
//            input = c.getClassLoader().getResourceAsStream(filename);
//        }

		try (BufferedReader br = new BufferedReader(new InputStreamReader(input))){

			while ((sCurrentLine = br.readLine()) != null) {
				
//				System.out.println("current line:\n\t" + sCurrentLine);

				if (sCurrentLine.contains("learnrate")) {
					learnrate = Double.parseDouble(sCurrentLine.split(" = ")[1]);
				}
				if (sCurrentLine.contains("# input neurons")) {
					nrInputNeurons = Integer.parseInt(sCurrentLine.split(" = ")[1]);
				}
				if (sCurrentLine.contains("# hidden neurons")) {
					nrHiddenNeurons = Integer.parseInt(sCurrentLine.split(" = ")[1]);
				}
				if (sCurrentLine.contains("# output neurons")) {
					nrOutputNeurons = Integer.parseInt(sCurrentLine.split(" = ")[1]);
					
					// also read the empty line
					sCurrentLine = br.readLine();
					readHiddenNeurons = true;
					
					// create input neurons
					for(int i = 0; i < nrInputNeurons; i++) {
						inputNeurons.add(new Neuron(hiddenNeurons, 0));
					}
					continue;
				}
				
				// create the hidden neurons
				if(readHiddenNeurons) {
					
					if(hiddenNeurons.size() >= nrHiddenNeurons) {
						readHiddenNeurons = false;
						readOutputNeurons = true;
						continue; // this consumes the empty line
					}
					weights = new double[nrInputNeurons];
					sWeights = sCurrentLine.split(",");
					
					// get the weights
					for (int i = 0; i < nrInputNeurons; i++) {
						weights[i] = Double.parseDouble(sWeights[i]);
					}
					bias = Double.parseDouble(sWeights[nrInputNeurons]);
					
					hiddenNeurons.add(new Neuron(outputNeurons, weights, bias));
				}
				
				// create the output neurons
				if(readOutputNeurons) {

					if(readOutputNeurons == false) {
						
						readOutputNeurons = true;
						readHiddenNeurons = false;
					}
					
					weights = new double[nrHiddenNeurons];
					sWeights = sCurrentLine.split(",");
					
					// get the weights
					for (int i = 0; i < nrHiddenNeurons; i++) {
						weights[i] = Double.parseDouble(sWeights[i]);
					}
					bias = Double.parseDouble(sWeights[nrHiddenNeurons]);
					
					outputNeurons.add(new Neuron(null, weights, bias));
				}
			}

		} catch (IOException e) {
			e.printStackTrace();
		}

		if(readOutputNeurons && learnrate != 0) {
			System.out.println("\nsuccessfully read the neural network from file " + filename);
			System.out.println("nrInputNeurons = " + nrInputNeurons + "\nnrHiddenNeurons = " + nrHiddenNeurons
					+ "\nnrOutputNeurons = " + nrOutputNeurons);
			
			// create the neural network
			Net net = new Net(inputNeurons, hiddenNeurons, outputNeurons, learnrate);
			return net;
		}
		
		System.out.println("readOutputNeurons = " + readOutputNeurons + " | learnrate = " + learnrate);
		System.err.println("something went wrong while reading the neural network from file " + filename);
		return null;
	}
}
