
public class DataPoint {
    double[] features;
    double label;

    public DataPoint(double[] features, double label) {
        this.features = features;
        this.label = label;
    }

    public double[] getFeatures() {
        return features;
    }

    public double getLabel() {
        return label;
    }

    public void setFeatures(double[] features) {
        this.features = features;
    }

    public void setLabel(double label) {
        this.label = label;
    }
}
