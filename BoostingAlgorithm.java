import edu.princeton.cs.algs4.Point2D;
import edu.princeton.cs.algs4.StdOut;

import java.util.LinkedList;


public class BoostingAlgorithm {

    private double[] weights; // weights
    private double[][] inputReduced; // reduced k-dim input
    private int[] labels; // labels
    private int n; // n
    private LinkedList<WeakLearner> weakLearners; // saves weaklearners
    private Clustering cl; // performs clustering

    // create clusters and initialize data structures
    public BoostingAlgorithm(double[][] input, int[] labels, Point2D[]
            locations, int k) {
        validate(input);
        validate(input[0]);
        validate(labels);
        validate(locations);

        n = input.length;

        if (labels.length != n) {
            throw new IllegalArgumentException("");
        }

        // create a clustering object
        cl = new Clustering(locations, k);

        // reduce dimensions
        inputReduced = new double[n][k];
        for (int i = 0; i < n; i++) {
            inputReduced[i] = cl.reduceDimensions(input[i]);
        }

        weakLearners = new LinkedList<WeakLearner>();

        // initialize weights
        weights = new double[n];
        for (int i = 0; i < n; i++) {
            weights[i] = 1.0 / n;
        }

        // assign other instance variables
        this.labels = labels;

    }


    // return the current weights
    public double[] weights() {
        return this.weights;
    }

    // apply one step of the boosting algorithm
    public void iterate() {
        // create a weak learner using current weights and the input
        WeakLearner wl = new WeakLearner(inputReduced, weights, labels);
        for (int i = 0; i < n; i++) {
            double inputResult = wl.predict(inputReduced[i]);
            if (inputResult != labels[i]) {
                // double the weight to make the algo predict
                // harder in future iterations
                weights[i] *= 2;
            }
        }

        weights = renormalize(weights);
        weakLearners.add(wl);
    }

    // re-normalizes an array. input = double[] output = double[]
    private double[] renormalize(double[] input) {
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += input[i];
        }
        double[] output = new double[n];
        for (int i = 0; i < n; i++) {
            output[i] = input[i] / sum;
        }
        return output;
    }

    // return the prediction of the learner for a new sample
    public int predict(double[] sample) {
        double[] reducedSample = cl.reduceDimensions(sample);
        int countZero = 0;
        int countOne = 0;
        for (WeakLearner wl : weakLearners) {
            if (wl.predict(reducedSample) == 1) countOne++; // output is 1
            else countZero++; // output is 0
        }

        if (countZero >= countOne) return 0; // return 0 in case of tie
        return 1; // else return 1

    }

    // helper method to check if the argument passed is null.
    private void validate(Object obj) {
        if (obj == null) throw new IllegalArgumentException("");
    }

    // unit testing
    public static void main(String[] args) {
        // read in the terms from a file
        DataSet training = new DataSet(args[0]);
        DataSet test = new DataSet(args[1]);
        int k = Integer.parseInt(args[2]);
        int iterations = Integer.parseInt(args[3]);

        // train the model
        BoostingAlgorithm model =
                new BoostingAlgorithm(training.input, training.labels,
                                      training.locations, k);
        for (int t = 0; t < iterations; t++)
            model.iterate();

        // calculate the training data set accuracy
        double trainingAccuracy = 0;
        for (int i = 0; i < training.n; i++)
            if (model.predict(training.input[i]) == training.labels[i])
                trainingAccuracy += 1;
        trainingAccuracy /= training.n;

        // calculate the test data set accuracy
        double testAccuracy = 0;
        for (int i = 0; i < test.n; i++)
            if (model.predict(test.input[i]) == test.labels[i])
                testAccuracy += 1;
        testAccuracy /= test.n;

        StdOut.println("Training accuracy of model: " + trainingAccuracy);
        StdOut.println("Test accuracy of model:     " + testAccuracy);

        for (double weight : model.weights()) {
            StdOut.print(weight + " ");
        }

        StdOut.println();
    }
}
