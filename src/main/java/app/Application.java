package app;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

import java.io.FileNotFoundException;


@SpringBootApplication
@EnableScheduling
public class Application {

    public static void main(String[] args) throws FileNotFoundException {
        SpringApplication.run(Application.class);
    }

}

