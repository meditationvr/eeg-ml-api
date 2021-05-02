package app.model;

import com.datastax.driver.core.utils.UUIDs;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;

import java.io.Serializable;
import java.sql.Timestamp;
import java.util.*;

public class EEGInstructionData implements Serializable {

    private UUID pk;
    private Date date;
    private ArrayList<String> eeg;
    private Map<String, String> predictedlabels;
    private String userid;

    public EEGInstructionData(ArrayList<String> eeg, String userid) {
        this.pk = UUIDs.timeBased();
        this.date = new Timestamp(new Date().getTime());
        this.eeg = eeg;
        this.userid = userid;
        this.predictedlabels = new HashMap<>();
    }

    public String getPk() {
        return pk.toString();
    }

    public void setPk(UUID pk) {
        this.pk = pk;
    }

    public Timestamp getDate() {
        return new Timestamp(date.getTime());
    }

    public void setDate(Date date) {
        this.date = date;
    }

    public ArrayList<String> getEeg() {
        return eeg;
    }


    public Vector getEegVector() {
        double[] values = new double[eeg.size()];

        for (int i = 0; i < eeg.size(); ++i) {
            values[i] = Double.parseDouble(eeg.get(i));
        }
        return Vectors.dense(values);
    }

    public void setEeg(ArrayList<String> eeg) {
        this.eeg = eeg;
    }

    public String getUserid() {
        return userid;
    }

    public void setUserid(String userid) {
        this.userid = userid;
    }

    public Map<String, String> getPredictedlabels() {
        return predictedlabels;
    }

    public void setPredictedlabels(Map<String, String> predictedlabels) {
        this.predictedlabels = predictedlabels;
    }

    public String toString() {
        return "UserId: " + this.userid + ", DataPacketValue: " + this.eeg;
    }
}
