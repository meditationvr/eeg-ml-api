package app.model;

import java.io.Serializable;
import java.sql.Timestamp;
import java.util.Date;

public class EEGCalibration implements Serializable {

    private String calibrationid;
    private String userid;
    private Date startdate;
    private Date enddate;
    private Boolean modelsgenerated;

    public EEGCalibration(String calibrationid, String userid, Date startdate, Date enddate, Boolean modelsgenerated) {
        this.calibrationid = calibrationid;
        this.userid = userid;
        this.startdate = startdate;
        this.enddate = enddate;
        this.modelsgenerated = modelsgenerated;
    }

    public String getCalibrationid() {
        return calibrationid;
    }

    public void setCalibrationId(String calibrationid) {
        this.calibrationid = calibrationid;
    }

    public String getUserid() {
        return userid;
    }

    public void setUserid(String userid) {
        this.userid = userid;
    }

    public Timestamp getStartdate() {
        return new Timestamp(startdate.getTime());
    }

    public void setStartdate(Date startdate) {
        this.startdate = startdate;
    }

    public Timestamp getEnddate() {
        if (enddate == null) return null;
        return new Timestamp(enddate.getTime());
    }

    public void setEnddate(Date enddate) {
        this.enddate = enddate;
    }

    public Boolean getModelsgenerated() {
        return modelsgenerated;
    }

    public void setModelsgenerated(Boolean modelsgenerated) {
        this.modelsgenerated = modelsgenerated;
    }


}
