package app.service;

import app.model.EEGCalibration;

import app.model.MLModelsSingleton;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;

import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.springframework.stereotype.Service;
import scala.Serializable;
import scala.Tuple2;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Calendar;

@Service
public class EEGDataService implements Serializable{

    private static final Logger log = LoggerFactory.getLogger(EEGDataService.class);

    public void generateNaiveBayesClassifier(Dataset<Row> eegDataDF, EEGCalibration calibration) throws IOException {
        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(eegDataDF);

        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 6 distinct values are treated as continuous.
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("eegVector")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(6)
                .fit(eegDataDF);

        // Split the data into training and test sets (40% held out for testing)
        Dataset<Row>[] splits = eegDataDF.randomSplit(new double[]{0.6, 0.4}, 1234L);
        Dataset<Row> trainingData = splits[0].cache();
        Dataset<Row> testData = splits[1];

        NaiveBayes trainer = new NaiveBayes()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("indexedFeatures");

        // Convert indexed labels back to original labels.
        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(MLModelsSingleton.getInstance().labels);

        // Chain indexers and forest in a Pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {labelIndexer, featureIndexer, trainer, labelConverter});

        // Train model. This also runs the indexers.
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions.
        Dataset<Row> predictions = model.transform(testData);

        // Print Metrics
        printMetrics(predictions, "nb", calibration.getUserid());

        // Select example rows to display.
        predictions.select("predictedLabel", "label", "eeg").show(5);

        // Select (prediction, true label) and compute test error
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

        NaiveBayesModel nb = (NaiveBayesModel)(model.stages()[2]);

        nb.save(getModelPath(calibration, "nb"));

        MLModelsSingleton.getInstance().naiveBayesClassifiers.put(calibration.getUserid(), nb);
    }


    public void generateMultilayerPerceptronClassifier(Dataset<Row> eegDataDF, EEGCalibration calibration) throws IOException {

        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(eegDataDF);

        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 6 distinct values are treated as continuous.
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("eegVector")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(6)
                .fit(eegDataDF);

        // Split the data into training and test sets (30% held out for testing)
        Dataset<Row>[] splits = eegDataDF.randomSplit(new double[]{0.6, 0.4}, 1234L);
        Dataset<Row> trainingData = splits[0].cache();
        Dataset<Row> testData = splits[1];

        // specify layers for the neural network:
        // input layer of size 6 (features), two intermediate of size 5 and 4
        // and output of size 3 (classes)
        int[] layers = new int[] {6, 5, 4, MLModelsSingleton.getInstance().labels.length};

        // create the trainer and set its parameters
        MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
                .setLayers(layers)
                .setBlockSize(128)
                .setSeed(1234L)
                .setMaxIter(100)
                .setLabelCol("indexedLabel")
                .setFeaturesCol("indexedFeatures");

        // Convert indexed labels back to original labels.
        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(MLModelsSingleton.getInstance().labels);

        // Chain indexers and forest in a Pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {labelIndexer, featureIndexer, trainer, labelConverter});

        // Train model. This also runs the indexers.
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions.
        Dataset<Row> predictions = model.transform(testData);

        // Print Metrics
        printMetrics(predictions, "mlp", calibration.getUserid());

        // Select example rows to display.
        predictions.select("predictedLabel", "label", "eeg").show(5);

        // Select (prediction, true label) and compute test error
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

        MultilayerPerceptronClassificationModel mlPerceptron = (MultilayerPerceptronClassificationModel)(model.stages()[2]);

        mlPerceptron.save(getModelPath(calibration, "mlp"));

        MLModelsSingleton.getInstance().mlPerceptronClassifiers.put(calibration.getUserid(), mlPerceptron);
    }

    public void generateRandomForestClassifier(Dataset<Row> eegDataDF, EEGCalibration calibration) throws IOException {
        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(eegDataDF);


        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 6 distinct values are treated as continuous.
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("eegVector")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(6)
                .fit(eegDataDF);

        // Split the data into training and test sets (40% held out for testing)
        Dataset<Row>[] splits = eegDataDF.randomSplit(new double[] {0.6, 0.4});
        Dataset<Row> trainingData = splits[0].cache();
        Dataset<Row> testData = splits[1];

        // Train a RandomForest model.
        RandomForestClassifier trainer = new RandomForestClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("indexedFeatures")
                .setNumTrees(20);

        // Convert indexed labels back to original labels.
        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(MLModelsSingleton.getInstance().labels);

        // Chain indexers and forest in a Pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {labelIndexer, featureIndexer, trainer, labelConverter});

        // Train model. This also runs the indexers.
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions.
        Dataset<Row> predictions = model.transform(testData);

        // Print Metrics
        printMetrics(predictions, "rf", calibration.getUserid());

        // Select example rows to display.
        predictions.select("predictedLabel", "label", "eeg").show(5);

        // Select (prediction, true label) and compute test error
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

        RandomForestClassificationModel rfModel = (RandomForestClassificationModel)(model.stages()[2]);

        rfModel.save(getModelPath(calibration, "rf"));

        MLModelsSingleton.getInstance().randomForestClassifiers.put(calibration.getUserid(), rfModel);
    }

    // Multiclass Metrics
    // https://spark.apache.org/docs/2.4.0/api/java/org/apache/spark/mllib/evaluation/MulticlassMetrics.html
    // https://github.com/ragnar-lothbrok/spark-demo/blob/master/src/main/java/com/spark/lograthmicregression/ClickThroughRateAnalytics.java
    @SuppressWarnings("rawtypes")
    private void printMetrics(Dataset<Row> predictions, String model, String userId) {

        predictions = predictions.select("predictedLabel", "indexedLabel");


        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = predictions.javaRDD().map(new Function<Row, Tuple2<Object, Object>>() {
            private static final long serialVersionUID = 1L;

            @Override
            public Tuple2<Object, Object> call(Row v1) {
                String predictedLabel = v1.get(0).toString();

                String predictedLabelNum = "0.0";
                if (predictedLabel.equals("Shallow")) {
                    predictedLabelNum = "0.0";
                } else if (predictedLabel.equals("Medium")) {
                    predictedLabelNum = "1.0";
                } else if (predictedLabel.equals("Deep")){
                    predictedLabelNum = "2.0";
                }
                //System.out.println("Double predictedLabelNum "  + predictedLabelNum + "\n");

                String indexedLabelNum = v1.get(1).toString();

                //System.out.println("Double indexedLabelNum "  + indexedLabelNum + "\n");
                return new Tuple2<>(Double.parseDouble(predictedLabelNum), Double.parseDouble(indexedLabelNum));
            }
        });


        // Obtain metrics
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());

        BufferedWriter writer = null;
        try {
            //create a temporary file
            String timeLog = new SimpleDateFormat("yyyyMMdd_HHmmss").format(Calendar.getInstance().getTime()) + "-" + userId + "-" + model + ".txt";
            File logFile = new File(timeLog); // C:\Users\Daniel\Desktop\workspace\MeditationVR\ml.txt

            // This will output the full path where the file will be written to...
            System.out.println(logFile.getCanonicalPath());

            writer = new BufferedWriter(new FileWriter(logFile));
            // Metrics
            writer.write("----------------\n");
            writer.write("Model: " + model + "\n");
            writer.write("----------------\n");
            // Confusion matrix
            Matrix confusion = metrics.confusionMatrix();
            writer.write("Shallow: 0.0\n");
            writer.write("Medium: 1.0\n");
            writer.write("Deep: 2.0\n\n");

            writer.write("Confusion matrix: \n" + confusion.toString() + "\n");

            // Overall statistics
            writer.write("Accuracy = " + metrics.accuracy() + "\n");

            // Stats by labels
            for (int i = 0; i < metrics.labels().length; i++) {
                writer.write("Class " + metrics.labels()[i] + " precision = " + metrics.precision(
                        metrics.labels()[i]) + "\n");
                writer.write("Class " + metrics.labels()[i] + " recall = " + metrics.recall(
                        metrics.labels()[i]) + "\n");
                writer.write("Class " + metrics.labels()[i] + " F1 score = " + metrics.fMeasure(
                        metrics.labels()[i]) + "\n");
                writer.write("Class " + metrics.labels()[i] + " False Positive Rate = " + metrics.falsePositiveRate(
                        metrics.labels()[i]) + "\n");
                writer.write("Class " + metrics.labels()[i] + " True Positive Rate = " + metrics.truePositiveRate(
                        metrics.labels()[i]) + "\n");
            }

            //Weighted stats
            writer.write("Weighted precision = " + metrics.weightedPrecision() + "\n");
            writer.write("Weighted recall = " + metrics.weightedRecall() + "\n");
            writer.write("Weighted F1 score = " + metrics.weightedFMeasure() + "\n");
            writer.write("Weighted false positive rate = " + metrics.weightedFalsePositiveRate() + "\n");
            writer.write("Weighted true positive rate = " + metrics.weightedTruePositiveRate() + "\n");

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                // Close the writer regardless of what happens...
                writer.close();
            } catch (Exception e) {
            }
        }
    }

    private String getModelPath(EEGCalibration calibration, String modelType) throws IOException {
        // Create and clear the user models directory
        Path modelsDirectoryPath = Paths.get(System.getenv("MODELS_PATH"), calibration.getCalibrationid() + "-" + calibration.getUserid());
        File modelsDirectory = modelsDirectoryPath.toFile();
        if (!modelsDirectory.exists()) {
            modelsDirectory.mkdir();
        }
        // FileUtils.cleanDirectory(modelsDirectory);

        // Save filter for future use while classifying instances
        Path modelPath = Paths.get(modelsDirectoryPath.toString(), "model-" + modelType);

        return modelPath.toString();
    }
}
