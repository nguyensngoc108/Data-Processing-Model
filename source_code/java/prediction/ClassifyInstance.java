package weka.api;

//import required classes
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

public class ClassifyInstance{
	public static void main(String args[]) throws Exception{
		
		// Load test data
        DataSource testSource = new DataSource("D:\\Year 4\\Data Mining\\Project\\data\\sales_test.arff");
        Instances testDataset = testSource.getDataSet();
        testDataset.setClassIndex(testDataset.numAttributes() - 1);

        // Load the saved model: OneR
        Classifier oneR = (Classifier) SerializationHelper.read("OneR.model");

        // Perform predictions and print actual class and OneR predicted class
        System.out.println("===================");
        System.out.println("Actual Class, OneR Predicted");
        for (int i = 0; i < testDataset.numInstances(); i++) {
            // Get class double value for current instance
            double actualValue = testDataset.instance(i).classValue();

            // Get Instance object of current instance
            Instance newInst = testDataset.instance(i);

            // Call classifyInstance, which returns a double value for the class
            double predOneR = oneR.classifyInstance(newInst);

            System.out.println(actualValue + ", " + predOneR);
        }
        
        // Evaluate the model on the test data
        Evaluation eval_oneR = new Evaluation(testDataset);
        eval_oneR.evaluateModel(oneR, testDataset);
        System.out.println("===================");
        System.out.println("Evaluation Results:");
        System.out.println(eval_oneR.toSummaryString());
        System.out.println(eval_oneR.toMatrixString());
        System.out.println(eval_oneR.toClassDetailsString());
        
        
     // Load the saved model: NaiveBayes
        Classifier nb = (Classifier) SerializationHelper.read("NaiveBayes.model");

        // Perform predictions and print actual class and NaiveBayes predicted class
        System.out.println("===================");
        System.out.println("Actual Class, Naivebayes Predicted");
        for (int i = 0; i < testDataset.numInstances(); i++) {
            // Get class double value for current instance
            double actualValue = testDataset.instance(i).classValue();

            // Get Instance object of current instance
            Instance newInst = testDataset.instance(i);

            // Call classifyInstance, which returns a double value for the class
            double predNB = nb.classifyInstance(newInst);

            System.out.println(actualValue + ", " + predNB);
        }
        
        // Evaluate the model on the test data
        Evaluation eval_nb = new Evaluation(testDataset);
        eval_nb.evaluateModel(nb, testDataset);
        System.out.println("===================");
        System.out.println("Evaluation Results:");
        System.out.println(eval_nb.toSummaryString());
        System.out.println(eval_nb.toMatrixString());
        System.out.println(eval_nb.toClassDetailsString());
        
        
        // Load the saved model: NaiveBayes
        Classifier rf = (Classifier) SerializationHelper.read("RandomForest.model");

        // Perform predictions and print actual class and RandomForest predicted class
        System.out.println("===================");
        System.out.println("Actual Class, RandomForest Predicted");
        for (int i = 0; i < testDataset.numInstances(); i++) {
            // Get class double value for current instance
            double actualValue = testDataset.instance(i).classValue();

            // Get Instance object of current instance
            Instance newInst = testDataset.instance(i);

            // Call classifyInstance, which returns a double value for the class
            double predRF = rf.classifyInstance(newInst);

            System.out.println(actualValue + ", " + predRF);
        }
        
        // Evaluate the model on the test data
        Evaluation eval_rf = new Evaluation(testDataset);
        eval_rf.evaluateModel(rf, testDataset);
        System.out.println("===================");
        System.out.println("Evaluation Results:");
        System.out.println(eval_rf.toSummaryString());
        System.out.println(eval_rf.toMatrixString());
        System.out.println(eval_rf.toClassDetailsString());
        
	}
}