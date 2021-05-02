package app.model;

import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.linalg.Vector;

import java.util.HashMap;
import java.util.Map;

// Java program implementing Singleton class
// with getInstance() method
public class MLModelsSingleton
{
    // static variable single_instance of type Singleton
    private static MLModelsSingleton single_instance = null;

    public String[] labels = {"Shallow", "Medium", "Deep"};
    public Map<String, RandomForestClassificationModel> randomForestClassifiers = new HashMap<>();
    public Map<String, MultilayerPerceptronClassificationModel> mlPerceptronClassifiers = new HashMap<>();
    public Map<String, NaiveBayesModel> naiveBayesClassifiers = new HashMap<>();

    public Map<String, String> classify(String userId, Vector eeg) {
        Map<String, String> predictedLabels = new HashMap<>();
        if (randomForestClassifiers.containsKey(userId)) {
            RandomForestClassificationModel randomForestClassifier = randomForestClassifiers.get(userId);
            Double predictionIndex = randomForestClassifier.predict(eeg);

            String prediction = labels[predictionIndex.intValue()];
            predictedLabels.put("RandomForest", prediction);
        } else {
            predictedLabels.put("RandomForest", "Generating Model");
        }

        if (mlPerceptronClassifiers.containsKey(userId)) {
            MultilayerPerceptronClassificationModel mlPerceptronClassifier = mlPerceptronClassifiers.get(userId);
            Double predictionIndex = mlPerceptronClassifier.predict(eeg);

            String prediction = labels[predictionIndex.intValue()];
            predictedLabels.put("MLPerceptron", prediction);
        } else {
            predictedLabels.put("MLPerceptron", "Generating Model");
        }

        if (naiveBayesClassifiers.containsKey(userId)) {
            NaiveBayesModel nbClassifier = naiveBayesClassifiers.get(userId);
            Double predictionIndex = nbClassifier.predict(eeg);

            String prediction = labels[predictionIndex.intValue()];
            predictedLabels.put("NaiveBayes", prediction);
        } else {
            predictedLabels.put("NaiveBayes", "Generating Model");
        }
        return predictedLabels;
    }

    // private constructor restricted to this class itself
    private MLModelsSingleton() {}

    // static method to create instance of Singleton class
    public static MLModelsSingleton getInstance()
    {
        if (single_instance == null)
            single_instance = new MLModelsSingleton();

        return single_instance;
    }
}