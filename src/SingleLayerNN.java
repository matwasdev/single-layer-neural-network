import javax.swing.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class SingleLayerNN {

    public static int numLanguages = 3;
    public static LinkedList<Perceptron> perceptrons = new LinkedList<>();

    public static List<DataPoint> trainingData = new LinkedList<>();
    public static List<DataPoint> testData = new LinkedList<>();

    public static double testSizeRatio =0.2;
    public static TreeMap<Integer, String> languageLabelsMap;

    public static void main(String[] args) throws IOException {
        Scanner scanner = new Scanner(System.in);

        initialize(scanner);
        loadAndPrepareData(scanner);
        trainModel();
        testModel();
        classificationLoop(scanner);
    }

    private static void initialize(Scanner scanner) {
        System.out.println("Enter the number of possible languages for classification:");
        numLanguages = Integer.parseInt(scanner.nextLine());
        languageLabelsMap = new TreeMap<>();
    }

    private static void loadAndPrepareData(Scanner scanner) throws IOException {
        collectLanguagePaths();
        System.out.println("Language label mapping: " + languageLabelsMap);
        System.out.println();

        System.out.println("Enter the test dataset size (e.g., 0.2):");
        testSizeRatio = Double.parseDouble(scanner.nextLine());

        Collections.shuffle(trainingData);
        splitTrainAndTestData();
    }

    private static void trainModel() {
        train();
        System.out.println("===================== TRAINING COMPLETED =====================");
        System.out.println();
    }

    private static void testModel() {
        test();
        System.out.println("===================== TESTING COMPLETED =====================");
        System.out.println();
    }

    private static void classificationLoop(Scanner scanner) {
        while (true) {
            System.out.println("1. Classify custom input text");
            System.out.println("2. Exit program");

            String input = scanner.nextLine();

            if (input.equals("1")) {
                classifyUserText(scanner);
            } else if (input.equals("2")) {
                break;
            }
        }
    }

    private static void classifyUserText(Scanner scanner) {
        System.out.println("Enter the text to classify (the longer the text, the better the result): ");
        String text = scanner.nextLine();

        String[] lines = { text };
        double[] lettersOccur = new double[26];
        calculateLetterOccurrences(lines, lettersOccur);
        normalizeVector(lettersOccur);

        TreeMap<Double, Integer> netMap = new TreeMap<>();
        for (Perceptron perceptron : perceptrons) {
            double activation = perceptron.calculateActivation(lettersOccur) - perceptron.threshold;
            netMap.put(activation, perceptron.languageLabel);
        }

        int predictedLabel = netMap.lastEntry().getValue();
        System.out.println("NET outputs: " + netMap);
        System.out.println("CLASSIFIED LANGUAGE: " + predictedLabel + " - " + languageLabelsMap.get(predictedLabel));
        System.out.println();
    }


    public static void train(){

        for(int i = 0; i< numLanguages; i++){
            int languageLabelForPerceptron = i;
            Perceptron langPerceptron = new Perceptron(trainingData,languageLabelForPerceptron);
            perceptrons.add(langPerceptron);
            langPerceptron.train();
        }
    }


    public static void test(){
        System.out.println("=====================TESTING=====================");
        int hits=0;

        for(DataPoint dataPoint : testData){

            int labelThatShouldBe = (int) dataPoint.getLabel();

            TreeMap<Double, Integer> netMap = new TreeMap<>();

            for(Perceptron perceptron : perceptrons){
                double currentNet = perceptron.calculateActivation(dataPoint.getFeatures()) - perceptron.threshold;
                netMap.put(currentNet, perceptron.languageLabel);
            }

            System.out.println("True language: " + labelThatShouldBe + " - " + languageLabelsMap.get(labelThatShouldBe));
            System.out.println("NET Answers: " + netMap);
            System.out.println("CLASSIFIED LANGUAGE AS: " + netMap.lastEntry().getValue() + " - " + languageLabelsMap.get(netMap.lastEntry().getValue()));
            System.out.println();

            if(netMap.lastEntry().getValue()  ==  labelThatShouldBe){
                ++hits;
            }
        }
        double accuracyTest = (double) hits / testData.size();
        System.out.println("TEST SIZE: " + testData.size());
        System.out.println("ACCURATE CLASSIFICATIONS: " + hits);
        System.out.println("Accuracy on TEST: " + accuracyTest);
    }


    public static void collectLanguagePaths() throws IOException {
        Scanner scanner = new Scanner(System.in);

        for(int i = 0; i< numLanguages; i++) {
            System.out.println("Enter language name for : "+ i);
            String language = scanner.nextLine();

            languageLabelsMap.put(i, language);

            Path langDirPath = chooseDirectory("Select language directory for: " + language);

            convertDirectoryFilesToVectors(langDirPath,i);
        }
    }

    private static Path chooseDirectory(String title) {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle(title);
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

        int result = chooser.showOpenDialog(null);
        if (result != JFileChooser.APPROVE_OPTION) {
            throw new RuntimeException("Not a proper catalog");
        }

        return chooser.getSelectedFile().toPath();
    }


    public static void convertDirectoryFilesToVectors(Path langDirPath, double languageLabel) throws IOException {
        Files.walk(langDirPath).filter(Files::isRegularFile).forEach(file -> {
            try {
                String[] lines = Files.readAllLines(file).toArray(new String[0]);
                DataPoint dataPoint = new DataPoint(new double[26], languageLabel);

                calculateLetterOccurrences(lines,dataPoint.getFeatures());
                normalizeVector(dataPoint.getFeatures());

                trainingData.add(dataPoint);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }


    public static void calculateLetterOccurrences(String[] lines, double[] lettersVector) {
        for(int i = 0; i<lines.length; i++){
            lines[i] = lines[i].toLowerCase();

            for(int j = 0; j<lines[i].length(); j++){

                if(lines[i].charAt(j) >= 'a' && lines[i].charAt(j) <= 'z'){
                    lettersVector[lines[i].charAt(j) - 'a']++;
                }
            }
        }
    }


    public static void normalizeVector(double[] vector) {
        if (vector.length == 0)
            throw new RuntimeException("Vector is empty");

        double d =0;
        for(int i =0; i<vector.length; i++){
            d += Math.pow(vector[i], 2);
        }

        d = Math.sqrt(d);

        if(d==0)
            throw new RuntimeException("Cannot normalize with zero");

        for(int i =0; i<vector.length; i++){
            vector[i] /= d;
        }
    }

    public static void splitTrainAndTestData(){
        int size =  (int) (testSizeRatio * trainingData.size());

        testData = new ArrayList<>(trainingData.subList(0, size));
        trainingData = new ArrayList<>(trainingData.subList(size, trainingData.size()));

        System.out.println("Training dataset size: " + trainingData.size());
        System.out.println("Test dataset size: " + testData.size());
    }
}
