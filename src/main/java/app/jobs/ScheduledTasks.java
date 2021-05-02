package app.jobs;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.*;

import app.model.EEGCalibration;
import app.model.EEGCalibrationData;
import app.model.EEGInstructionData;
import app.model.MLModelsSingleton;
import app.service.EEGDataService;

import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static com.datastax.spark.connector.japi.CassandraJavaUtil.*;
import static com.datastax.spark.connector.japi.CassandraJavaUtil.javaFunctions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
public class ScheduledTasks {

    private static final Logger log = LoggerFactory.getLogger(ScheduledTasks.class);

    private final SimpleDateFormat dateFormat = new SimpleDateFormat("HH:mm:ss.SSS");

    private final String sparkMaster = System.getenv("SPARK_MASTER");
    private final String sparkMasterPort = System.getenv("SPARK_MASTER_PORT");

    private final String dbHost = System.getenv("DB_HOST");
    private final String dbPort = System.getenv("DB_PORT");
    private final String dbUsername = System.getenv("DB_USERNAME");
    private final String dbPassword = System.getenv("DB_PASSWORD");
    private final String dbKeyspaceName = System.getenv("KEYSPACE_NAME");

    private final String localExecution = System.getenv("LOCAL_EXECUTION");

    private SparkSession spark;

    @Autowired
    private EEGDataService eegDataService;

    @Scheduled(cron = "*/60 * * * * *" )
    //@Scheduled(fixedDelay=5000)
    public void generateClassifiers() {
        log.info("Generating Classifiers at {}", dateFormat.format(new Date()));

        if (localExecution != null)
            System.setProperty("hadoop.home.dir", "C:\\winutils\\");

        SparkConf sparkConf = new SparkConf();

        if (localExecution != null) {
            sparkConf.setMaster("local[*]");
        } else {
            sparkConf.setMaster("spark://" + sparkMaster + ":" + sparkMasterPort);
            String[] jars = new String[1];
            jars[0] = "/usr/local/service/target/eeg-ml-api-0.1.0.jar";
            sparkConf.setJars(jars);
        }

        sparkConf.setAppName("EEGClassificationJob")
                .set("spark.driver.allowMultipleContexts", "true")
                .set("spark.cassandra.connection.host", dbHost)
                .set("spark.cassandra.connection.port", dbPort)
                .set("spark.cassandra.auth.username", dbUsername)
                .set("spark.cassandra.auth.password", dbPassword);

        spark = SparkSession
                .builder().config(sparkConf)
                .getOrCreate();

        JavaRDD<EEGCalibration> userCalibrationsRDD = javaFunctions(spark.sparkContext())
            .cassandraTable(dbKeyspaceName, "user_calibrations", mapRowTo(EEGCalibration.class));

        List<EEGCalibration> userCalibrations = new ArrayList<>();

        userCalibrationsRDD.collect().forEach(calibration -> {
                // If calibration in progress or models already generated
                if (calibration.getEnddate() == null || calibration.getModelsgenerated())
                    return;

                JavaRDD<EEGCalibrationData> userCalibrationsEegDataRDD = javaFunctions(spark.sparkContext())
                        .cassandraTable(dbKeyspaceName, "user_calibrations_eeg_data", mapRowTo(EEGCalibrationData.class))
                        .where("calibrationId = ?", calibration.getCalibrationid());

                Dataset<Row> userCalibrationsEegDataDF = spark.createDataFrame(userCalibrationsEegDataRDD, EEGCalibrationData.class);

                userCalibrationsEegDataDF.show(10);

                if (userCalibrationsEegDataDF.count() == 0)
                    return;

                try {
                    eegDataService.generateRandomForestClassifier(userCalibrationsEegDataDF, calibration);
                    eegDataService.generateMultilayerPerceptronClassifier(userCalibrationsEegDataDF, calibration);
                    eegDataService.generateNaiveBayesClassifier(userCalibrationsEegDataDF, calibration);
                    calibration.setModelsgenerated(true);
                    userCalibrations.add(calibration);
                } catch (Exception e) {
                    Path modelsDirectoryPath = Paths.get(System.getenv("MODELS_PATH"), calibration.getCalibrationid() + "-" + calibration.getUserid());
                    File modelsDirectory = modelsDirectoryPath.toFile();
                    if (modelsDirectory.exists()) {
                        try {
                            FileUtils.cleanDirectory(modelsDirectory);
                        } catch (IOException e1) {
                            e1.printStackTrace();
                        }
                    }
                    log.info("Failed to generated classifiers!");
                    log.info(e.getMessage());
                    e.printStackTrace();
                }
        });

        JavaRDD<EEGCalibration> eegCalibrationRDD = new JavaSparkContext(spark.sparkContext()).parallelize(userCalibrations);

        if (eegCalibrationRDD.count() > 0) {
            javaFunctions(eegCalibrationRDD)
                    .writerBuilder(dbKeyspaceName, "user_calibrations",
                            mapToRow(EEGCalibration.class))
                    .saveToCassandra();
        }

        // Update non-classified user instructions eeg data
        JavaRDD<EEGInstructionData> userInstructionsEegDataRDD = javaFunctions(spark.sparkContext())
                .cassandraTable(dbKeyspaceName, "user_instructions_eeg_data", mapRowTo(EEGInstructionData.class))
                .where("predictedlabels CONTAINS ?", "Generating Model")
                .map(userInstructionEegData -> {
                    userInstructionEegData.setPredictedlabels(
                            MLModelsSingleton.getInstance()
                                    .classify(userInstructionEegData.getUserid(), userInstructionEegData.getEegVector())
                    );
                    return userInstructionEegData;
                });

        if (userInstructionsEegDataRDD.count() > 0) {
            javaFunctions(userInstructionsEegDataRDD)
                    .writerBuilder(dbKeyspaceName, "user_instructions_eeg_data",
                            mapToRow(EEGInstructionData.class))
                    .saveToCassandra();
        }
    }
}
