import edu.princeton.cs.algs4.StdOut;

public class WeakLearner {

    private int dimension; // dimension of predictor
    private double value; // value of predictor
    private int sign; // sign of predictor
    private int k; // k


    // train the weak leaner
    public WeakLearner(double[][] input, double[] weights, int[] labels) {
        validate(input);
        validate(input[0]);
        validate(weights);
        validate(labels);

        if (input.length != weights.length) throw new IllegalArgumentException("");
        if (input.length != labels.length) throw new IllegalArgumentException("");

        int n = input.length;
        this.k = input[0].length;

        for (int i = 0; i < n; i++) {
            if (weights[i] < 0.0 || !(labels[i] == 0 || labels[i] == 1)) {
                throw new IllegalArgumentException("");
            }
        }

        double champ = -1;

        for (int currSign = 0; currSign <= 1; currSign++) {
            for (int currDim = 0; currDim < k; currDim++) {
                for (int currVal = 0; currVal < n; currVal++) {
                    double refVal = input[currVal][currDim];

                    double totalWeight = 0.0;
                    for (int currPoint = 0; currPoint < n; currPoint++) {
                        int currLabel = 0;
                        if (currSign == 0 && input[currPoint][currDim] > refVal) {
                            currLabel = 1;
                        }
                        if (currSign == 1 && input[currPoint][currDim] <= refVal) {
                            currLabel = 1;
                        }
                        if (labels[currPoint] == currLabel) {
                            totalWeight += weights[currPoint];
                        }
                    }

                    if (totalWeight >= champ) {
                        champ = totalWeight;
                        dimension = currDim;
                        sign = currSign;
                        value = refVal;
                    }
                }
            }
        }
    }

    // return the prediction of the weaklearner for a new sample
    public int predict(double[] sample) {
        if (sample == null || sample.length != k) {
            throw new IllegalArgumentException("");
        }
        if ((sign == 0 && sample[dimension] <= value) ||
                (sign == 1 && sample[dimension] > value)) {
            return 0;
        }
        else if ((sign == 0 && sample[dimension] > value) ||
                (sign == 1 && sample[dimension] <= value)) {
            return 1;
        }
        return -1; // unexpected error. should not return
    }

    // return the dimension used by the weaklearner to separate data
    public int dimensionPredictor() {
        return dimension;
    }

    // return the value used by the weaklerner to separate data
    public double valuePredictor() {
        return value;
    }

    // return the sign the learner uses to separate the data
    public int signPredictor() {
        return sign;
    }

    // helper method to check if the argument passed is null.
    private void validate(Object obj) {
        if (obj == null) throw new IllegalArgumentException("");
    }

    // unit testing (required)
    public static void main(String[] args) {
        DataSet training = new DataSet(args[0]);
        DataSet test = new DataSet(args[1]);

        double[] weights = new double[800];
        for (int i = 0; i < 800; i++) {
            weights[i] = 1.0;
        }

        // train the model
        WeakLearner model = new WeakLearner(training.input, weights,
                                            training.labels);


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

        StdOut.println("Dimension predictor: " + model.dimensionPredictor());
        StdOut.println("Value predictor: " + model.valuePredictor());
        StdOut.println("Sign predictor: " + model.signPredictor());
    }

}
