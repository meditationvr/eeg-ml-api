package app.model;

import com.datastax.driver.core.utils.UUIDs;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;

import java.io.Serializable;

import java.sql.Timestamp;

import java.util.ArrayList;
import java.util.Date;
import java.util.UUID;

public class EEGCalibrationData implements Serializable {

    private UUID pk;
    private Date date;
    private ArrayList<String> eeg;
    private String label;
    private String userid;
    private String calibrationid;


    public EEGCalibrationData(ArrayList<String> eeg, String label, String userid, String calibrationid) {
        this.pk = UUIDs.timeBased();
        this.date = new Timestamp(new Date().getTime());
        this.eeg = eeg;
        this.userid = userid;
        this.label = label;
        this.calibrationid = calibrationid;
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

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String getUserid() {
        return userid;
    }

    public void setUserid(String userid) {
        this.userid = userid;
    }

    public String getCalibrationid() {
        return calibrationid;
    }

    public void setCalibrationid(String calibrationid) {
        this.calibrationid = calibrationid;
    }

    public String toString() {
        return "CalibrationId: " + this.calibrationid + ", DataPacketValue: " + this.eeg;
    }


}
