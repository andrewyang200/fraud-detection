Programming Assignment 7: Fraud Detection

/* *****************************************************************************
 *  Describe how you implemented the Clustering constructor
 **************************************************************************** */
First, we created an EdgeWeightedGraph of all possible pairs of vertices. This was
done by using a double for loop to find all possible pairs of vertices and then for
each pair of vertices that are not the same, we create a new edge with the weight
being the distance between the two vertices and inserted it into the EdgeWeightedGraph.
Afterward, we called the built-in KruskalMST algorithm to find the minimum spanning
tree of the EdgeWeightedGraph by considering on the smallest m-k edges. This insures
that the new graph we created is the MST of the main EdgeWeightedGraph. And then we
updated our instance variable graph to store the MST we found.

/* *****************************************************************************
 *  Describe how you implemented the WeakLearner constructor
 **************************************************************************** */

We first set the champion weight to be -1. Then, we iterated through both signs
(0 & 1), all possible dimension (0 to k), the possible values given the input
points, and each of the points themselves. Afterward, we assigned an arbitrary
label at first say 0. We then checked to see based on the sign we currently have
and whether the current point if less than or greater than the value. If we had
the opposite label, we change the label to 1. At the same time, we also keep a
running sum of the total number of correct points we identified. If the label
the weaklearner chooses lines up with the corresponding label given, we add the
weight of the point to the running sum. In the end, we check to see if the running
sum of the weights is greater than out champion weight. If so, we update the
all instance variables to be the current iteration of dimension, value, and sign,
as well as the champion weights. Doing so ensures that we go through all possible
combinations of dp, dv, and ds.


/* *****************************************************************************
 *  Consider the large_training.txt and large_test.txt datasets.
 *  Run the boosting algorithm with different values of k and T (iterations),
 *  and calculate the test data set accuracy and plot them below.
 *
 *  (Note: if you implemented the constructor of WeakLearner in O(kn^2) time
 *  you should use the training.txt and test.txt datasets instead, otherwise
 *  this will take too long)
 **************************************************************************** */

      k          T         test accuracy       time (seconds)
   --------------------------------------------------------------------------
      5 1000 0.8 0.328
      10 1000 0.8 0.715
      30 1000 0.97 1.68
      50 1000 0.86 2.62
      30 10000 0.97 15.74
      30 5000 0.97 8.16
      35 5000 0.93 10.04
      25 5000 0.90 6.602
      30 6000 0.97 9.6
      30 6500 0.97 10.02



/* *****************************************************************************
 *  Find the values of k and T that maximize the test data set accuracy,
 *  while running under 10 second. Write them down (as well as the accuracy)
 *  and explain:
 *   1. Your strategy to find the optimal k, T.
 *   2. Why a small value of T leads to low test accuracy.
 *   3. Why a k that is too small or too big leads to low test accuracy.
 **************************************************************************** */

1. We tried finding the optimal k first. We found that around 30 is the best.
Then, we increased T as to maximize the training given the optimal clustering
It turned out that 30 is the best K, and the most training we can get in
10 seconds is 6000. The accuracy came out to around 0.97.


2. A small value of T means the model is not well-trained enough. There are not
enough iterations for new decision stumps to come in and be trained. This means
the instances in which old stumps perform poorly aren't being paid enough
attention to and rectified.

3. A k that is too small leads to over generalization as we might group points
that aren't closely related into the same cluster. This will cause the algorithm
so over simplify a lot of the classification and not consider the points within
its respective context.

A k that is too large means that we are considering each point as itself. This
may lead to overfitting as our algorithm would consider each individual point.
Therefore, if there is slight deviations from what the algorithm has already seen,
then it predict the wrong results.

/* *****************************************************************************
 *  Known bugs / limitations.
 **************************************************************************** */
There are no known bugs or limitations.

/* *****************************************************************************
 *  Describe any serious problems you encountered.
 **************************************************************************** */
Our knlogn doesn't work. Here is our best attempt: Logically it should work,
but for some reason the testing is just not right. There are not enough test-
cases to debug against on Tigerfile, which could have helped.

public class WeakLearner {

    private int dimension; // dimension of predictor
    private double value; // value of predictor
    private int sign; // sign of predictor
    private int k; // k


    // train the weak leaner
    public WeakLearner(double[][] input, double[] weights, int[] labels) {
        if (input == null || input[0] == null
                || weights == null || labels == null) {
            throw new IllegalArgumentException("");
        }


        int n = weights.length;
        k = input[0].length;
        if (input.length != n || labels.length != n) {
            throw new IllegalArgumentException("");
        }
        // check
        for (int i = 0; i < n; i++) {
            if (weights[i] < 0 || !(labels[i] == 0 || labels[i] == 1)) {
                throw new IllegalArgumentException("");
            }
        }

        double maxScore = Double.NEGATIVE_INFINITY;

        // iterate through each dimension - k
        for (int currDim = 0; currDim < k; currDim++) {
            DataPoint[] dataPoints = new DataPoint[n];
            for (int i = 0; i < n; i++) {
                dataPoints[i] = new DataPoint(
                        input[i][currDim], labels[i], weights[i]);
            }

            // Sort data points based on their value
            Arrays.sort(dataPoints);

            double weightSignZero = 0;
            double weightSignOne = 0;

            double threshold = dataPoints[0].value;

            for (int i = 0; i < n; i++) { // pass through to find values for
                // when thresh is at the very bottom value

                // initial thresh
                if (i == 0) {
                    if (dataPoints[i].label == 1) // first one red
                        weightSignOne += dataPoints[i].weight; // count red
                    else // first one blue
                        weightSignZero += dataPoints[i].weight; // count blue

                }

                if (dataPoints[i].label == 0) {// if it's blue, count  red
                    weightSignOne += dataPoints[i].weight;
                }
                else
                    weightSignZero += dataPoints[i].weight;

            }

            maxScore = Math.max(weightSignZero, weightSignOne);
            this.dimension = currDim;
            this.value = threshold;
            if (weightSignZero > weightSignOne) this.sign = 0;
            else this.sign = 1;


            // iterate thru thresholds in ascending order
            for (int i = 1; i < n; i++) {
                threshold = dataPoints[i].value;
                // weight sign one
                if (dataPoints[i].label == 1) { // if it's red
                    weightSignOne += dataPoints[i].weight; // one more correct
                }
                else { // if it's blue
                    weightSignZero += dataPoints[i].weight; // one more correct
                }

                double score = Math.max(weightSignZero, weightSignOne);

                if (score > maxScore) {
                    maxScore = score;
                    // this.dimension = currDim;
                    this.value = threshold;
                    if (weightSignZero > weightSignOne) this.sign = 0;
                    else this.sign = 1;
                }

            }
        }
    }

    // helper class to allow sorting absed on value
    private static class DataPoint implements Comparable<DataPoint> {
            double value;
            int label;
            double weight;

            // constructor
            public DataPoint(double value, int label, double weight) {
                this.value = value;
                this.label = label;
                this.weight = weight;
            }

            // compares based on value
            public int compareTo(DataPoint other) {
                return Double.compare(this.value, other.value);
            }
        }




/* *****************************************************************************
 *  List any other comments here. Feel free to provide any feedback
 *  on how much you learned from doing the assignment, and whether
 *  you enjoyed doing it.
 **************************************************************************** */

frustrating to get the knlogn part working and lack of support too.
