
public class Main {

	public static void main(String[] args) {

		// new net with sensors {food, water, danger, strongerThanDanger} and output
		// {run}
		Netz net = new Netz(4, 2, 1, 0.1);

		double[][] trainingInput = new double[][] { { 0, 0, 0, 1 }, { 1, 0, 0, 1 }, { 0, 1, 0, 1 }, { 0, 0, 1, 0 },
				{ 1, 0, 1, 0 }, { 0, 1, 1, 0 } };
		double[][] trainingOutput = new double[][] { { 0 }, { 0 }, { 0 }, { 1 }, { 1 }, { 1 } };

		for (int j = 0; j < 100; j++) {
			
			for (int i = 0; i < trainingInput.length; i++) {

				System.out.println("training data " + i);
				net.train(trainingInput[i], trainingOutput[i], 10000);
			}

			System.out.println("\nprediction for {0, 0, 0, 1} is " + net.compute(new double[] { 0, 0, 0, 1 })[0]);
			System.out.println("prediction for {1, 1, 1, 0} is " + net.compute(new double[] { 1, 1, 1, 0 })[0] + "\n");
			
			System.out.println("prediction for {1, 1, 1, 0} is " + net.compute(new double[] { 1, 1, 1, 0 })[0]);
			System.out.println("prediction for {1, 1, 1, 1} is " + net.compute(new double[] { 1, 1, 1, 1 })[0]);
			System.out.println("prediction for {1, 1, 0, 1} is " + net.compute(new double[] { 1, 1, 0, 1 })[0] + "\n");
		}
	}

}
