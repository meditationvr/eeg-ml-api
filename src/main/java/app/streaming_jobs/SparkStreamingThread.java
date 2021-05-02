package app.streaming_jobs;

import app.model.EEGCalibration;
import app.model.EEGInstructionData;
import app.model.MLModelsSingleton;
import app.service.EEGDataService;
import app.model.EEGCalibrationData;

import com.google.gson.*;

import org.apache.spark.*;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.streaming.api.java.*;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.kafka010.ConsumerStrategies;
import org.apache.spark.streaming.kafka010.KafkaUtils;
import org.apache.spark.streaming.kafka010.LocationStrategies;

import static com.datastax.spark.connector.japi.CassandraJavaUtil.*;
import static com.datastax.spark.connector.japi.CassandraStreamingJavaUtil.javaFunctions;
import com.datastax.spark.connector.japi.CassandraStreamingJavaUtil;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.common.serialization.StringDeserializer;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.PostConstruct;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

@Component
public class SparkStreamingThread extends Thread {

    private static final Logger log = LoggerFactory.getLogger(SparkStreamingThread.class);

    private final String localExecution = System.getenv("LOCAL_EXECUTION");

    private final String sparkMaster = System.getenv("SPARK_MASTER");
    private final String sparkMasterPort = System.getenv("SPARK_MASTER_PORT");

    private final String kafkaBroker = System.getenv("KAFKA_BROKER");
    private final String kafkaBrokerPort = System.getenv("KAFKA_BROKER_PORT");

    private final String dbHost = System.getenv("DB_HOST");
    private final String dbPort = System.getenv("DB_PORT");
    private final String dbUsername = System.getenv("DB_USERNAME");
    private final String dbPassword = System.getenv("DB_PASSWORD");
    private final String dbKeyspaceName = System.getenv("KEYSPACE_NAME");

    private final SparkConf sparkConf = new SparkConf();

    @Autowired
    private EEGDataService eegDataService;

    @PostConstruct
    public void init() {
        start();
    }

    public void run() {

        try {

            if (localExecution != null)
                System.setProperty("hadoop.home.dir", "C:\\winutils\\");

            SparkConf sparkConf = new SparkConf();

            if (localExecution != null) {
                sparkConf.setMaster("local[2]");
            } else {
                sparkConf.setMaster("spark://" + sparkMaster + ":" + sparkMasterPort);
                String[] jars = new String[1];
                jars[0] = "/usr/local/service/target/eeg-ml-api-0.1.0.jar";
                sparkConf.setJars(jars);
            }


            sparkConf.setAppName("EEGStreamingJob")
                    .set("spark.driver.allowMultipleContexts", "true")
                    .set("spark.cassandra.connection.host", dbHost)
                    .set("spark.cassandra.connection.port", dbPort)
                    .set("spark.cassandra.auth.username", dbUsername)
                    .set("spark.cassandra.auth.password", dbPassword);

            JavaStreamingContext streamingContext = new JavaStreamingContext(sparkConf, Durations.seconds(1));

            streamingContext.sparkContext().setLogLevel("ERROR");

            pullMLModelsToMemory(streamingContext);
            saveCalibrationEEGData(streamingContext);
            classifyInstructionEEGData(streamingContext);

            // Start the computation
            streamingContext.start();
            streamingContext.awaitTermination();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void pullMLModelsToMemory(JavaStreamingContext streamingContext) {
        // Push ML models to drivers memory
        CassandraStreamingJavaUtil.javaFunctions(streamingContext)
            .cassandraTable(dbKeyspaceName, "user_calibrations", mapRowTo(EEGCalibration.class))
            .where("modelsGenerated = ?", true)
            .collect()
            .forEach(calibration -> {
                Path modelPath = Paths.get(System.getenv("MODELS_PATH"), calibration.getCalibrationid() + "-" + calibration.getUserid());
                Path rfPath = Paths.get(modelPath.toString(), "model-rf");
                MLModelsSingleton.getInstance().randomForestClassifiers.put(
                        calibration.getUserid(), RandomForestClassificationModel.load(rfPath.toString())
                );
                Path mlpPath = Paths.get(modelPath.toString(), "model-mlp");
                MLModelsSingleton.getInstance().mlPerceptronClassifiers.put(
                        calibration.getUserid(), MultilayerPerceptronClassificationModel.load(mlpPath.toString())
                );
                Path nbPath = Paths.get(modelPath.toString(), "model-nb");
                MLModelsSingleton.getInstance().naiveBayesClassifiers.put(
                        calibration.getUserid(), NaiveBayesModel.load(nbPath.toString())
                );
            });
    }

    private void classifyInstructionEEGData(JavaStreamingContext streamingContext) {

        String kafkaInstructionTopic = System.getenv("KAFKA_INSTRUCTION_TOPIC");

        Map<String, Object> kafkaParams = new HashMap<>();
        kafkaParams.put("bootstrap.servers", (kafkaBroker + ":" + kafkaBrokerPort));
        kafkaParams.put("key.deserializer", StringDeserializer.class);
        kafkaParams.put("value.deserializer", StringDeserializer.class);
        kafkaParams.put("group.id", (kafkaInstructionTopic + "_stream"));
        kafkaParams.put("auto.offset.reset", "earliest"); // "latest"
        kafkaParams.put("enable.auto.commit", false);

        Collection<String> topics = Arrays.asList(kafkaInstructionTopic);

        JavaInputDStream<ConsumerRecord<String, String>> messages  =
                KafkaUtils.createDirectStream(
                        streamingContext,
                        LocationStrategies.PreferConsistent(),
                        ConsumerStrategies.Subscribe(topics, kafkaParams)
                );

        JavaDStream<EEGInstructionData> eegInstructionDataRows = messages
                .map(record -> {
                    JsonElement jElement = new JsonParser().parse(record.value());
                    JsonObject jObject = jElement.getAsJsonObject();

                    JsonArray jsonArr = jObject.getAsJsonArray("DataPacketValue");
                    String userId = jObject.get("userId").getAsString();

                    ArrayList<String> eeg = (ArrayList<String>) new Gson().fromJson(jsonArr, ArrayList.class);

                    EEGInstructionData eegInstructionData = new EEGInstructionData(eeg, userId);

                    eegInstructionData.setPredictedlabels(
                            MLModelsSingleton.getInstance().classify(userId, eegInstructionData.getEegVector())
                    );

                    return eegInstructionData;
                });

        CassandraStreamingJavaUtil.javaFunctions(eegInstructionDataRows)
                .writerBuilder(dbKeyspaceName, "user_instructions_eeg_data",
                        mapToRow(EEGInstructionData.class))
                .saveToCassandra();
    }

    private void saveCalibrationEEGData(JavaStreamingContext streamingContext) {

        String kafkaCalibrationTopic = System.getenv("KAFKA_CALIBRATION_TOPIC");

        Map<String, Object> kafkaParams = new HashMap<>();
        kafkaParams.put("bootstrap.servers", (kafkaBroker + ":" + kafkaBrokerPort));
        kafkaParams.put("key.deserializer", StringDeserializer.class);
        kafkaParams.put("value.deserializer", StringDeserializer.class);
        kafkaParams.put("group.id", (kafkaCalibrationTopic + "_stream"));
        kafkaParams.put("auto.offset.reset", "earliest"); // "latest"
        kafkaParams.put("enable.auto.commit", false);

        Collection<String> topics = Arrays.asList(kafkaCalibrationTopic);

        JavaInputDStream<ConsumerRecord<String, String>> messages  =
                KafkaUtils.createDirectStream(
                        streamingContext,
                        LocationStrategies.PreferConsistent(),
                        ConsumerStrategies.Subscribe(topics, kafkaParams)
                );

        JavaDStream<EEGCalibrationData> eegCalibrationDataRows = messages
                    .map(record -> {
                        JsonElement jElement = new JsonParser().parse(record.value());
                        JsonObject jObject = jElement.getAsJsonObject();

                        JsonArray jsonArr = jObject.getAsJsonArray("DataPacketValue");

                        ArrayList<String> eeg = (ArrayList<String>) new Gson().fromJson(jsonArr, ArrayList.class);

                        return new EEGCalibrationData(
                                eeg,
                                jObject.get("Label").getAsString(),
                                jObject.get("userId").getAsString(),
                                jObject.get("calibrationId").getAsString()
                        );
                    });

        javaFunctions(eegCalibrationDataRows)
                .writerBuilder(dbKeyspaceName, "user_calibrations_eeg_data",
                        mapToRow(EEGCalibrationData.class))
                .saveToCassandra();
    }
}
