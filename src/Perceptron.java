import java.util.*;


public class Perceptron {

    public  List<DataPoint> trainingData = new ArrayList<>();
    public double learningRate = 0.5;
    public double threshold = 1;
    public double[] weights;
    public int languageLabel;


    Perceptron(List<DataPoint> trainData, int languageLabel) {
        this.languageLabel = languageLabel;

        trainingData = new ArrayList<>();

        for (DataPoint dp : trainData) {
            DataPoint dataPoint = new DataPoint(Arrays.copyOf(dp.getFeatures(), dp.getFeatures().length), dp.getLabel());

            trainingData.add(dataPoint);
        }
    }


    public void train(){
        System.out.println();
        System.out.println("Training started for PERCEPTRON: " + languageLabel);
        setWeights();
        for(int i =0; i<trainingData.size(); i++){
            if(trainingData.get(i).getLabel() != languageLabel){
                trainingData.get(i).setLabel(0);
            }
            else
                trainingData.get(i).setLabel(1);
        }

        double accuracy = 0;
        int epoch=0;

        while(epoch<5){
            accuracy=trainPerceptron();
            System.out.println("current accuracy:"+ accuracy);
            System.out.println("epoch: "+ epoch++);
            System.out.println();
        }
        System.out.println("FINAL ACCURACY: "+accuracy);


        System.out.println();
        System.out.println("Training COMPLETED for PERCEPTRON: " + languageLabel);
        System.out.println();
    }




    private double trainPerceptron(){
        boolean shouldAdjust=false;
        int hits=0;

        double[] prevVector = new double[trainingData.getFirst().getFeatures().length];
        int prevD=0;
        int prevY=0;

        for (DataPoint dataPoint : trainingData) {
            if (shouldAdjust) {
                adjustWeightsAndThreshold(prevVector, prevD, prevY);
                shouldAdjust = false;
            }


            double activation = calculateActivation(dataPoint.getFeatures());


            int d = (int) dataPoint.getLabel();
            int y = activation >= threshold ? 1 : 0;


            if (d == y) {
                hits++;
            } else {
                prevVector=dataPoint.getFeatures();
                prevD=d;
                prevY=y;
                shouldAdjust = true;
            }

            System.out.println(Arrays.toString(dataPoint.getFeatures()));
            System.out.println("Expected : " + d);
            System.out.println("Predicted  : " + y);
            System.out.println();
        }

        return (double)hits/ trainingData.size();
    }



    public double calculateActivation(double[] v1){
        return dotProduct(v1, weights);
    }



    private void adjustWeightsAndThreshold(double[] x1, int d, int y){
        double[] adjustedWeights = new double[weights.length];

        int dMinY = (d-y);
        for(int i =0; i<x1.length; i++){
            adjustedWeights[i] = x1[i]*dMinY* learningRate;
        }
        for (int i = 0; i < adjustedWeights.length; i++) {
            adjustedWeights[i] = weights[i]+adjustedWeights[i];
        }

        weights =adjustedWeights;
        threshold = threshold - (dMinY* learningRate);

        System.out.println("Adjusted weights: " + Arrays.toString(adjustedWeights));
        System.out.println("New threshold: " + threshold +'\n');
    }



    private void setWeights(){
        weights = new double[trainingData.getFirst().getFeatures().length];
        for(int i = 0; i < trainingData.getFirst().getFeatures().length; i++) {
            double rand = -0.1 + (0.2 * Math.random());
            weights[i] = rand;
        }
        System.out.println("Initial weights: " + Arrays.toString(weights));
    }


    public double dotProduct(double[] v1, double[] v2){
        double result = 0;

        if(v1.length != v2.length)
            throw new RuntimeException("Vectors are not the same size");

        for(int i =0; i<v1.length; i++){
            result += v1[i]*v2[i];
        }

        return result;
    }

}
