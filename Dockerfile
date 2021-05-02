FROM adoptopenjdk/openjdk8
MAINTAINER Daniel Marchena (danielmapar@gmail.com)

ENV SPARK_MASTER spark-master
ENV SPARK_MASTER_PORT 7077
ENV DB_HOST match-making-db
ENV DB_USERNAME match_making_user
ENV DB_PASSWORD match_making_pw
ENV KEYSPACE_NAME match_making
ENV DB_PORT 9042
ENV KAFKA_BROKER kafka-broker
ENV KAFKA_BROKER_PORT 9092
ENV KAFKA_CALIBRATION_TOPIC eeg-calibration
ENV KAFKA_INSTRUCTION_TOPIC eeg-instruction
ENV MODELS_PATH /resources/classifiers

RUN apt-get update
RUN apt-get install -y maven
RUN apt-get install -y libc6 libc6-dbg libc6-dev
RUN mkdir /resources && mkdir /resources/classifiers && chmod 777 /resources && chmod 777 /resources/*
COPY pom.xml /usr/local/service/pom.xml
COPY src /usr/local/service/src
COPY wait-for-it.sh /usr/local/service/wait-for-it.sh
WORKDIR /usr/local/service
RUN mvn package

ENTRYPOINT chmod 777 ./wait-for-it.sh && ./wait-for-it.sh -t 120 match-making-db:9042 && ./wait-for-it.sh -t 120 kafka-broker:9092 && ./wait-for-it.sh -t 120 spark-master:7077 && sleep 140 && java -jar target/eeg-ml-api-0.1.0.jar
