# fraud-detection

## Clustering Constructor Implementation

The `Clustering` constructor is implemented as follows:

1. **Graph Construction**: 
   - An `EdgeWeightedGraph` is created containing all possible pairs of vertices.
   - A double for-loop iterates over all vertices to generate pairs.
   - For each distinct pair, an edge is created with a weight equal to the Euclidean distance between the vertices, and added to the graph.

2. **Minimum Spanning Tree (MST)**:
   - The Kruskal’s MST algorithm is invoked on the `EdgeWeightedGraph`, considering only the smallest `m - k` edges (where `m` is the number of edges and `k` is the desired number of clusters).
   - This ensures the resulting graph is the MST of the original graph.

3. **Update**:
   - The instance variable `graph` is updated to store the computed MST.

## WeakLearner Constructor Implementation

The `WeakLearner` constructor is implemented as follows:

1. **Initialization**: 
   - The champion weight (best score) is initialized to `-1`.

2. **Exhaustive Search**:
   - Iterate over:
     - Both signs (`0` and `1`),
     - All dimensions (`0` to `k-1`),
     - All possible threshold values derived from input points,
     - Each point in the dataset.
   - For each combination:
     - Assign an initial label (e.g., `0`).
     - Adjust the label based on the sign and whether the point’s value is less than or greater than the threshold (e.g., flip to `1` if conditions suggest).
     - Compute a running sum of weights for correctly classified points (if the predicted label matches the given label, add the point’s weight).

3. **Optimization**:
   - If the running sum exceeds the current champion weight, update the instance variables: `dimension`, `value`, `sign`, and the champion weight.
   - This ensures the best combination of dimension, value, and sign is selected after exploring all possibilities.

## Boosting Algorithm Performance

The boosting algorithm was tested with the `large_training.txt` and `large_test.txt` datasets for various values of `k` (clusters) and `T` (iterations). Results are below:

| k   | T     | Test Accuracy | Time (seconds) |
|-----|-------|---------------|----------------|
| 5   | 1000  | 0.80          | 0.328          |
| 10  | 1000  | 0.80          | 0.715          |
| 30  | 1000  | 0.97          | 1.68           |
| 50  | 1000  | 0.86          | 2.62           |
| 30  | 10000 | 0.97          | 15.74          |
| 30  | 5000  | 0.97          | 8.16           |
| 35  | 5000  | 0.93          | 10.04          |
| 25  | 5000  | 0.90          | 6.602          |
| 30  | 6000  | 0.97          | 9.6            |
| 30  | 6500  | 0.97          | 10.02          |

## Optimal k and T

### Optimal Values
- **k = 30**
- **T = 6000**
- **Test Accuracy = 0.97**
- **Time = 9.6 seconds**

### Analysis
1. **Strategy**:
   - First, tested various `k` values with a fixed `T` (e.g., 1000) to identify a peak accuracy (around `k = 30`).
   - Then, fixed `k = 30` and increased `T` to maximize accuracy while staying under 10 seconds, settling on `T = 6000`.

2. **Small T Impact**:
   - A small `T` limits iterations, preventing sufficient training. New decision stumps cannot adequately address errors from prior stumps, leading to underfitting and low accuracy.

3. **k Extremes**:
   - **Too Small k**: Overgeneralization occurs, grouping unrelated points into clusters, oversimplifying classification and ignoring context.
   - **Too Large k**: Overfitting results, as each point is treated individually, making the model sensitive to minor deviations in test data.

## Known Bugs / Limitations

There are no known bugs or limitations.

## Serious Problems Encountered

The `O(kn log n)` implementation of `WeakLearner` does not work as expected. Despite a logically sound approach, testing revealed issues. Limited test cases on Tigerfile hindered debugging. Below is the best attempt:

```java
public class WeakLearner {
    private int dimension; // dimension of predictor
    private double value;  // value of predictor
    private int sign;      // sign of predictor
    private int k;         // number of dimensions

    public WeakLearner(double[][] input, double[] weights, int[] labels) {
        if (input == null || input[0] == null || weights == null || labels == null) {
            throw new IllegalArgumentException("");
        }

        int n = weights.length;
        k = input[0].length;
        if (input.length != n || labels.length != n) {
            throw new IllegalArgumentException("");
        }

        for (int i = 0; i < n; i++) {
            if (weights[i] < 0 || !(labels[i] == 0 || labels[i] == 1)) {
                throw new IllegalArgumentException("");
            }
        }

        double maxScore = Double.NEGATIVE_INFINITY;

        for (int currDim = 0; currDim < k; currDim++) {
            DataPoint[] dataPoints = new DataPoint[n];
            for (int i = 0; i < n; i++) {
                dataPoints[i] = new DataPoint(input[i][currDim], labels[i], weights[i]);
            }

            Arrays.sort(dataPoints);

            double weightSignZero = 0;
            double weightSignOne = 0;
            double threshold = dataPoints[0].value;

            for (int i = 0; i < n; i++) {
                if (i == 0) {
                    if (dataPoints[i].label == 1) weightSignOne += dataPoints[i].weight;
                    else weightSignZero += dataPoints[i].weight;
                }
                if (dataPoints[i].label == 0) weightSignOne += dataPoints[i].weight;
                else weightSignZero += dataPoints[i].weight;
            }

            maxScore = Math.max(weightSignZero, weightSignOne);
            this.dimension = currDim;
            this.value = threshold;
            this.sign = (weightSignZero > weightSignOne) ? 0 : 1;

            for (int i = 1; i < n; i++) {
                threshold = dataPoints[i].value;
                if (dataPoints[i].label == 1) weightSignOne += dataPoints[i].weight;
                else weightSignZero += dataPoints[i].weight;

                double score = Math.max(weightSignZero, weightSignOne);
                if (score > maxScore) {
                    maxScore = score;
                    this.value = threshold;
                    this.sign = (weightSignZero > weightSignOne) ? 0 : 1;
                }
            }
        }
    }

    private static class DataPoint implements Comparable<DataPoint> {
        double value;
        int label;
        double weight;

        public DataPoint(double value, int label, double weight) {
            this.value = value;
            this.label = label;
            this.weight = weight;
        }

        public int compareTo(DataPoint other) {
            return Double.compare(this.value, other.value);
        }
    }
}
