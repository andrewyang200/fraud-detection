import edu.princeton.cs.algs4.CC;
import edu.princeton.cs.algs4.Edge;
import edu.princeton.cs.algs4.EdgeWeightedGraph;
import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.KruskalMST;
import edu.princeton.cs.algs4.Point2D;
import edu.princeton.cs.algs4.StdOut;

public class Clustering {

    private int k; // number of clusters
    private int m; // number of vertices
    private EdgeWeightedGraph graph; // reduced cluster graph

    // run the clustering algorithm and create the clusters
    public Clustering(Point2D[] locations, int k) {
        validate(locations);
        if (k < 1 || k > locations.length) throw new IllegalArgumentException("");

        this.k = k;
        this.m = locations.length;

        graph = new EdgeWeightedGraph(m);

        // create an edge-weighted graph connecting each pair of vertices
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                if (locations[i] == null || locations[j] == null)
                    throw new IllegalArgumentException("");
                if (j != i) {
                    Edge e = new Edge(i, j, locations[i].distanceTo(locations[j]));
                    graph.addEdge(e);
                }
            }
        }

        // compute the MST
        KruskalMST mst = new KruskalMST(graph);
        Iterable<Edge> edges = mst.edges();

        // create a new graph with m vertices and m-k edges
        // this represents the cluster graph
        EdgeWeightedGraph newGraph = new EdgeWeightedGraph(m);

        int counter = 0;
        for (Edge e : edges) {
            if (counter == m - k) break;
            newGraph.addEdge(e);
            counter++;
        }

        // update the instance variable
        graph = newGraph;
    }

    // return the cluster of the ith point
    public int clusterOf(int i) {
        if (i < 0 || i > m - 1) throw new IllegalArgumentException("");
        CC cc = new CC(graph);
        return cc.id(i);
    }

    // use the clusters to reduce the dimensions of an input
    public double[] reduceDimensions(double[] input) {
        validate(input);
        if (input.length != m) throw new IllegalArgumentException("");
        double[] result = new double[k];

        for (int i = 0; i < m; i++) {
            result[clusterOf(i)] += input[i];
        }
        return result;
    }

    // helper method to check if the argument passed is null.
    private void validate(Object obj) {
        if (obj == null) throw new IllegalArgumentException("");
    }

    // unit testing (required)
    public static void main(String[] args) {
        In in = new In("princeton_locations.txt");
        int len = in.readInt();
        Point2D[] locations = new Point2D[len];
        for (int i = 0; i < len; i++) {
            locations[i] = new Point2D(in.readDouble(), in.readDouble());
        }

        Clustering cl = new Clustering(locations, 5);

        StdOut.println(cl.clusterOf(2));

        double[] testArr = {
                5.0, 6.0, 7.0, 0.0, 6.0, 7.0, 5.0, 6.0, 7.0, 0.0,
                6.0, 7.0, 0.0, 6.0, 7.0, 0.0, 6.0, 7.0, 0.0, 6.0, 7.0
        };

        double[] reduced = cl.reduceDimensions(testArr);
        for (int i = 0; i < reduced.length; i++) {
            StdOut.print(reduced[i] + " ");
        }

        StdOut.println();
    }


}
