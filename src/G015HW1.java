import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class G015HW1
{
    public static void main(String[] args) throws IOException
    {

        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: 2 integers K and H, a string S, and a path to the file
        if (args.length != 4)
        {
            throw new IllegalArgumentException("USAGE: num_partitions H_value country file_path");
        }

        // SPARK SETUP
        SparkConf conf = new SparkConf(true).setAppName("Homework 1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN"); // Reduce number of log lines

        // INPUT READING
        int K = Integer.parseInt(args[0]);  // Read number of partitions
        int H = Integer.parseInt(args[1]);  // Read number of products with highest popularity
        String S = args[2];                 // Read path to the file storing the dataset

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 1
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read input file and subdivide it into K partitions
        JavaRDD<String> rawData = sc.textFile(args[3]).repartition(K).cache();
        // Print the number of rows read from the input file (number of elements of the RDD)
        long numrows = rawData.count(); // Number of rows
        System.out.println("Number of rows = " + numrows);

        // Declaring RDDs
        JavaPairRDD<String, Integer> productCustomer;
        JavaPairRDD<String, Integer> productPopularity1;
        JavaPairRDD<String, Integer> productPopularity2;

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 2
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        /*
         * TOKEN STRUCTURE
         * 0 -> TransactionID (a string uniquely representing a transaction),
         * 1 -> ProductID (a string uniquely representing a product),
         * 2 -> Description (a string describing the product),
         * 3 -> Quantity (an integer representing the units of product purchased),
         * 4 -> InvoiceDate (the date of the transaction),
         * 5 -> UnitPrice (a real representing the price of a unit of product),
         * 6 -> CustomerID (an integer uniquely representing a customer),
         * 7 -> Country (a string representing the country of the customer).
         */

        productCustomer = rawData
                // In the map phase we use the pair product-customer as key to guarantee
                // a constant local memory usage during the reduce phase
                .mapToPair((document) ->
                {   // <-- MAP PHASE (R1)
                    String[] tokens = document.split(","); // Array of String, split document content with comma
                    if (Long.parseLong(tokens[3]) > 0)
                    {
                        if (tokens[7].equals(S) || S.equals("all"))
                        {
                            // Add new pair with product-customer as key and true (valid pair) as value
                            return new Tuple2<>(new Tuple2<>(tokens[1], Integer.parseInt(tokens[6])), true);
                        }
                    }
                    // If the token is not valid we add a tuple as key (placeholder)
                    // and a value false ready to be filtered in the next step
                    return new Tuple2<>(new Tuple2<>("invalid", -1), false);
                })
                .filter(element -> element._2) //Remove invalid tuples
                .groupByKey() // <-- SHUFFLE+GROUPING
                // In the reduce phase we simply return a pair (key: productID, value: customerID) for every key (productID, customerID)
                .mapToPair((element) ->
                { // <-- REDUCE PHASE (R1), DELETE DUPLICATES
                    return new Tuple2<>(element._1._1, element._1._2); // product-customer pair
                });

        // Print number of pairs in the RDD productCustomer
        long paircount = productCustomer.count();
        System.out.println("Product-Customer Pairs = " + paircount);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 3
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        productPopularity1 = productCustomer
                .mapPartitionsToPair((element) ->
                {    // <-- REDUCE PHASE (R2A)
                    HashMap<String, Integer> counts = new HashMap<>();
                    while (element.hasNext())
                    {
                        Tuple2<String, Integer> tuple = element.next();
                        counts.put(tuple._1, 1 + counts.getOrDefault(tuple._1, 0));
                    }
                    ArrayList<Tuple2<String, Integer>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Integer> e : counts.entrySet())
                    {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                }).groupByKey() // <-- SHUFFLE+GROUPING
                .mapValues((it) ->
                { // <-- REDUCE PHASE (R3A)
                    int sum = 0;
                    for (int c : it)
                    {
                        sum += c;
                    }
                    return sum;
                });

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 4
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        productPopularity2 = productCustomer
                .mapToPair((element) ->
                { // <-- MAP PHASE (R2B)
                    // Since there are no duplicates (key,value) we can change
                    // CustomerID into 1 so the reduceByKey() method will compute the sum of
                    // different CustomerID for the specific product
                    return new Tuple2<>(element._1, 1);
                })
                .reduceByKey((x, y) -> x+y);    //<-- REDUCE PHASE (R2B)

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 5
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if(H > 0)
        {
            JavaPairRDD<Integer, String> sortedResults;
            sortedResults = productPopularity1
                    // In the map phase we invert key and values so that the popularity becomes the key,
                    // we then sort by popularity in not ascending order
                    .mapToPair((element) ->
                    { // <-- MAP PHASE
                        return new Tuple2<>(element._2,element._1);
                    }).sortByKey(false);

            // Prints the ProductID and Popularity of the H products with highest Popularity
            // We use the method take() instead of collect() to guarantee more efficiency
            System.out.println("Top " + H + " Products and their Popularities");
            for(Tuple2<Integer, String> line : sortedResults.take(H))
                System.out.print("Product " + line._2 + " Popularity " + line._1 + "; ");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // TASK 6
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if(H == 0)
        {
            // Print all the pairs of the two RDD productPopularity1 and productPopularity2
            // in increasing lexicographic order of ProductID
            System.out.println("productPopularity1:");
            for(Tuple2<String, Integer> line : productPopularity1.sortByKey().collect())
                System.out.print("Product: " + line._1 + " Popularity: " + line._2 + "; ");

            System.out.println("\nproductPopularity2:");
            for(Tuple2<String, Integer> line : productPopularity2.sortByKey().collect())
                System.out.print("Product: " + line._1 + " Popularity: " + line._2 + "; ");
        }
    }
}