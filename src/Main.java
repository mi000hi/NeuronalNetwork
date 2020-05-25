import NeuronalNetwork.Netz;
import Utils.DataReader;
import Utils.DataWriter;

public class Main {

	public static void main(String[] args) {

		// new net with sensors {food, water, danger, strongerThanDanger} and output
		// {run}
		Netz net = new Netz(4, 2, 1, 0.1);

		double[][] trainingInput = new double[][] { { 0, 0, 0, 1 }, { 1, 0, 0, 1 }, { 0, 1, 0, 1 }, { 0, 0, 1, 0 },
				{ 1, 0, 1, 0 }, { 0, 1, 1, 0 } };
		double[][] trainingOutput = new double[][] { { 0 }, { 0 }, { 0 }, { 1 }, { 1 }, { 1 } };

		net.train(trainingInput, trainingOutput, 1000000);

		System.out.println(
				"\nprediction for {0, 0, 0, 1} is (should be 0) " + net.compute(new double[] { 0, 0, 0, 1 })[0]);
		System.out.println(
				"prediction for {1, 1, 1, 0} is (should be 1) " + net.compute(new double[] { 1, 1, 1, 0 })[0] + "\n");

		System.out.println("prediction for {1, 1, 1, 0} is " + net.compute(new double[] { 1, 1, 1, 0 })[0]);
		System.out.println("prediction for {1, 1, 1, 1} is " + net.compute(new double[] { 1, 1, 1, 1 })[0]);
		System.out.println("prediction for {1, 1, 0, 1} is " + net.compute(new double[] { 1, 1, 0, 1 })[0] + "\n");
		
		// write net to file
		DataWriter.writeNetToFile("net.txt", net);
		
		// write new net from file
		Netz net2 = DataReader.readNetFromFile("net.txt");
		
		System.out.println("prediction for {1, 1, 0, 1} from net2 is " + net2.compute(new double[] { 1, 1, 0, 1 })[0] + "\n");
	}

}
