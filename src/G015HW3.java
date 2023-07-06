import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.lang.Math.sqrt;
import static java.util.Collections.sort;

public class G015HW3
{

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// MAIN PROGRAM 
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static void main(String[] args) throws Exception
    {
        if (args.length != 4)
        {
            throw new IllegalArgumentException("USAGE: filepath k z L");
        }

        // ----- Initialize variables
        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int z = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);
        long start, end; // variables for time measurements

        // ----- Set Spark Configuration
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkConf conf = new SparkConf(true).setAppName("MR k-center with outliers");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // ----- Read points from file
        start = System.currentTimeMillis();
        JavaRDD<Vector> inputPoints = sc.textFile(args[0], L)
                .map(x -> strToVector(x))
                .repartition(L)
                .cache();
        long N = inputPoints.count();
        end = System.currentTimeMillis();

        // ----- Print input parameters
        System.out.println("File : " + filename);
        System.out.println("Number of points N = " + N);
        System.out.println("Number of centers k = " + k);
        System.out.println("Number of outliers z = " + z);
        System.out.println("Number of partitions L = " + L);
        System.out.println("Time to read from file: " + (end - start) + " ms");

        // ---- Solve the problem
        ArrayList<Vector> solution = MR_kCenterOutliers(inputPoints, k, z, L);

        // ---- Compute the value of the objective function
        start = System.currentTimeMillis();
        double objective = computeObjective(inputPoints, solution, z);
        end = System.currentTimeMillis();
        System.out.println("Objective function = " + objective);
        System.out.println("Time to compute objective function: " + (end - start) + " ms");

    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// AUXILIARY METHODS
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method strToVector: input reading
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str)
    {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++)
        {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method euclidean: distance function
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double euclidean(Vector a, Vector b)
    {
        return Math.sqrt(Vectors.sqdist(a, b));
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method MR_kCenterOutliers: MR algorithm for k-center with outliers 
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> MR_kCenterOutliers(JavaRDD<Vector> points, int k, int z, int L)
    {
        long start1, end1, start2, end2;
        //------------- ROUND 1 ---------------------------
        start1 = System.currentTimeMillis();
        JavaRDD<Tuple2<Vector, Long>> coreset = points.mapPartitions(x ->
        {
            ArrayList<Vector> partition = new ArrayList<>();
            while (x.hasNext()) partition.add(x.next());
            ArrayList<Vector> centers = kCenterFFT(partition, k + z + 1);
            ArrayList<Long> weights = computeWeights(partition, centers);
            ArrayList<Tuple2<Vector, Long>> c_w = new ArrayList<>();
            for (int i = 0; i < centers.size(); ++i)
            {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weights.get(i));
                c_w.add(i, entry);
            }
            return c_w.iterator();
        }); // END OF ROUND 1

        //------------- ROUND 2 ---------------------------

        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>((k + z) * L);
        elems.addAll(coreset.collect());
        end1 = System.currentTimeMillis();
        //
        // ****** ADD YOUR CODE
        // ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
        // ****** Measure and print times taken by Round 1 and Round 2, separately
        // ****** Return the final solution
        //
        start2 = System.currentTimeMillis();
        ArrayList<Long> weights = new ArrayList<>();
        ArrayList<Vector> centers = new ArrayList<>();
        for (Tuple2<Vector, Long> i : elems)
        {
            weights.add(i._2);
            centers.add(i._1);
        }
        ArrayList<Vector> solution = SeqWeightedOutliers(centers, weights, k, z, 2);
        end2 = System.currentTimeMillis();
        //----- Print time needed for each round
        System.out.println("Time Round 1: " + (end1 - start1) + " ms");
        System.out.println("Time Round 2: " + (end2 - start2) + " ms");
        return solution;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method kCenterFFT: Farthest-First Traversal
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> kCenterFFT(ArrayList<Vector> points, int k)
    {

        final int n = points.size();
        double[] minDistances = new double[n];
        Arrays.fill(minDistances, Double.POSITIVE_INFINITY);

        ArrayList<Vector> centers = new ArrayList<>(k);

        Vector lastCenter = points.get(0);
        centers.add(lastCenter);
        double radius = 0;

        for (int iter = 1; iter < k; iter++)
        {
            int maxIdx = 0;
            double maxDist = 0;

            for (int i = 0; i < n; i++)
            {
                double d = euclidean(points.get(i), lastCenter);
                if (d < minDistances[i])
                {
                    minDistances[i] = d;
                }

                if (minDistances[i] > maxDist)
                {
                    maxDist = minDistances[i];
                    maxIdx = i;
                }
            }

            lastCenter = points.get(maxIdx);
            centers.add(lastCenter);
        }
        return centers;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeWeights: compute weights of coreset points
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Long> computeWeights(ArrayList<Vector> points, ArrayList<Vector> centers)
    {
        Long weights[] = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for (int i = 0; i < points.size(); ++i)
        {
            double tmp = euclidean(points.get(i), centers.get(0));
            int mycenter = 0;
            for (int j = 1; j < centers.size(); ++j)
            {
                if (euclidean(points.get(i), centers.get(j)) < tmp)
                {
                    mycenter = j;
                    tmp = euclidean(points.get(i), centers.get(j));
                }
            }
            // System.out.println("Point = " + points.get(i) + " Center = " + centers.get(mycenter));
            weights[mycenter] += 1L;
        }
        ArrayList<Long> fin_weights = new ArrayList<>(Arrays.asList(weights));
        return fin_weights;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method SeqWeightedOutliers: sequential k-center with outliers
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    //
    // ****** ADD THE CODE FOR SeqWeightedOuliers from HW2
    //
    public static ArrayList<Vector> SeqWeightedOutliers(ArrayList<Vector> P, ArrayList<Long> W, int k, int z, double alpha)
    {
        double r = Double.MAX_VALUE;
        double[][] distance_matrix = new double[P.size()][P.size()];
        // Since we use the Euclidean L2-distances between all the points in the set P many times,
        // we compute them only one time now and we memorize them in the matrix distance_matrix
        for (int i = 0; i < P.size(); i++)
        {
            for (int j = 0; j <= i; j++)
            {
                // We compute the Euclidean L2-distance between two points i and j
                // Since the distance between i and j is the same between j and i,
                // we leverage this fact, making the whole process more efficient
                distance_matrix[i][j] = Math.sqrt(Vectors.sqdist(P.get(i), P.get(j)));
                distance_matrix[j][i] = distance_matrix[i][j];
                // Compute minimum distance between first k+z+1 points,
                // necessary for the calculation of r
                if ((i < k + z + 1) && (j < i))
                {
                    if (distance_matrix[i][j] < r)
                    {
                        r = distance_matrix[i][j];
                    }
                }
            }
        }
        // Divide r by 2, so we get the value we are looking for
        r /= 2;

        // Print initial guess
        System.out.println("Initial guess = " + r);
        // Initialize variable to record how many times we update r
        int numberGuesses = 1;

        while (true)
        {
            // To use the values of the distances calculated before,
            // we need an array that keeps track of the elements
            // that are removed from the set Z at every iteration.
            // To do this we use the array covered, which is initialized to false since at the beginning no points are covered
            boolean[] covered = new boolean[P.size()];
            // All points that are not covered (outliers)
            ArrayList<Vector> Z = new ArrayList<>(P);
            // The set of centers (initially empty)
            ArrayList<Vector> S = new ArrayList<>();
            // Compute the total weights of all points in the set P
            long W_z = 0L;
            for (Long aWeight : W)
            {
                W_z += aWeight;
            }
            while ((S.size() < k) && (W_z > 0))
            {
                long max = 0L;
                int newcenter = 0;
                for (int i = 0; i < P.size(); i++) // i is the index of a point in the P set
                {
                    // Compute weights of ball-covered point
                    long ball_weight = 0L;
                    // Now we may have that one single point correspond to two different indexes,
                    // one for the set Z and the other for the set P
                    // Therefore the j index keeps track of its index in the Z set
                    // (so that we can know its weight and once that it is covered we can remove it from Z)
                    // and the x index keeps track of its index in the P set
                    // (so that we can calculate its distance from the selected point of index i (in P) using the matrix
                    // we calculated before)
                    for (int j = 0, x = 0; j < Z.size() && x < P.size(); j++, x++)
                    {
                        // Check if the value is present in the set Z
                        // If the value is not present, we continue iterating through the distance_matrix,
                        // without skipping any value of the set Z
                        if (covered[x])
                        {
                            j--; // Necessary to not skip next element in the set Z
                            continue;
                        }
                        // If the distance between the selected point in P and the selected point in Z
                        // is lower or equal to (1+2*alpha)*r, we add the weight of this last point from the set Z
                        // to the variable ball_weight
                        if (distance_matrix[i][x] <= (1 + 2 * alpha) * r)
                        {
                            ball_weight += W.get(x);
                        }
                    }
                    // We keep track of the maximum weight calculated before
                    // with respect to the maximum value already found
                    // So we save the position of this point of the set P,
                    // that will become the new center to be added in the set S
                    if (ball_weight > max)
                    {
                        max = ball_weight;
                        newcenter = i;
                    }
                }
                // Add the point found before to the set S
                S.add(P.get(newcenter));
                // Same indexes as in the previous for loop
                for (int j = 0, x = 0; j < Z.size() && x < P.size(); j++, x++)
                {
                    // Check if the value is present in the set Z
                    // If the value is not present, we continue iterating through the distance_matrix,
                    // without skipping any value of the set Z
                    if (covered[x])
                    {
                        j--; // Necessary to not skip next element in the set Z
                        continue;
                    }
                    // If the distance between the newcenter and the selected point in Z is lower or equal to (3+4*alpha)*r
                    // we delete this point from the set Z and we subtract its weight from W_z
                    if (distance_matrix[newcenter][x] <= (3 + 4 * alpha) * r)
                    {
                        W_z -= W.get(x);
                        Z.remove(j);
                        covered[x] = true; // Update the array, so we know that this point is no longer in the sets Z and W
                        j--; // Necessary for not skip next element in the set Z
                    }
                }
            }
            // The algorithm stops at the smallest guess r
            // for which the points of Z (the outliers),
            // have total weight <= z,
            // so we return the set current centers S
            if (W_z <= z)
            {
                System.out.println("Final guess = " + r);
                System.out.println("Number of guesses = " + numberGuesses);
                return S;
            }
            // Otherwise, it restarts the algorithm with the doubled value of r (2r)
            else
            {
                r *= 2;
                numberGuesses++;
            }
        }
    }


// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeObjective: computes objective function  
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double computeObjective(JavaRDD<Vector> points, ArrayList<Vector> centers, int z)
    {

        //
        // ****** ADD THE CODE FOR computeObjective
        //
        JavaRDD<Double> rddDistance = points.mapPartitions(x ->
        {
            //ArrayList<Vector> partition = new ArrayList<>();
            ArrayList<Double> distances = new ArrayList<>();
            while (x.hasNext())
            {
                Vector point = x.next();
                double minimumDistance = Double.MAX_VALUE;
                // Compute distance d(x,S)
                for (Vector vector : centers)
                {
                    double tmpDistance = sqrt(Vectors.sqdist(point, vector));
                    // If x is closer to the current center then we update the distance
                    if (tmpDistance < minimumDistance)
                        minimumDistance = tmpDistance;
                }
                // Add the distance d(x,S) just computed to the list of all distances
                distances.add(minimumDistance);
            }
            return distances.iterator();
        });

        // Return the score of the k center after removing z elements

        List<Double> cost = rddDistance.top(z+1);
        return cost.get(z);
    }
}
