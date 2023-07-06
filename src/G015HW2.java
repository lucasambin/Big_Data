import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import static java.lang.Math.sqrt;
import static java.util.Collections.sort;

public class G015HW2
{
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
            ArrayList<Vector> Z = new ArrayList<Vector>(P);
            // The set of centers (initially empty)
            ArrayList<Vector> S = new ArrayList<Vector>();
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
                        W_z -= W.get(j);
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

    public static double ComputeObjective(ArrayList<Vector> P, ArrayList<Vector> S, int z)
    {
        ArrayList<Double> distances = new ArrayList<>();
        // Compute distance d(x,S) for every x in P
        for (Vector x : P)
        {
            double minimumDistance = Double.MAX_VALUE;
            // Compute distance d(x,S)
            for (Vector vector : S)
            {
                double tmpDistance = sqrt(Vectors.sqdist(x, vector));
                // If x is closer to the current center then we update the distance
                if (tmpDistance < minimumDistance)
                    minimumDistance = tmpDistance;
            }
            // Add the distance d(x,S) just computed to the list of all distances
            distances.add(minimumDistance);
        }
        // Sort all the distances
        sort(distances);
        // Return the score of the k center after removing z elements
        return (distances.get(distances.size() - z - 1));
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Input reading methods
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

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException
    {
        if (Files.isDirectory(Paths.get(filename)))
        {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }

    public static void main(String[] args) throws IOException
    {
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: <path_to_file>, number of centers, number of allowed outliers
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 3)
        {
            throw new IllegalArgumentException("USAGE: file_path num_centers num_outliers");
        }


        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int z = Integer.parseInt(args[2]);

        ArrayList<Vector> inputPoints = readVectorsSeq(filename);
        ArrayList<Long> weights = new ArrayList<>();

        int listSize = inputPoints.size();
        int i = 0;
        long unitWeight = 1L;
        while (i < listSize)
        {
            // Every point has weight 1
            weights.add(unitWeight);
            i++;
        }

        // First part of the output
        System.out.println("Input size n = " + listSize);
        System.out.println("Number of centers k = " + k);
        System.out.println("Number of outliers z =  " + z);

        // Run and record execution time of the method SeqWeightedOutliers()
        long startTime = System.nanoTime();
        ArrayList<Vector> solution = SeqWeightedOutliers(inputPoints, weights, k, z, 0);
        long endTime = System.nanoTime();
        long totalTime = (endTime - startTime) / 1000000;

        // Run method ComputeObjective()
        double objective = ComputeObjective(inputPoints, solution, z);

        // Last part of the output
        System.out.println("Objective function = " + objective);
        System.out.println("Time of SeqWeightedOutliers = " + totalTime);
    }
}