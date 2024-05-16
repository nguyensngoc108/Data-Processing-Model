package weka.api;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;

public class DataSplitting {
    public static void main(String[] args) throws Exception {
        
    	// Load data
        DataSource source = new DataSource("D:\\Year 4\\Data Mining\\Project\\data\\dimension_reduction_sales_clothes.arff");
        Instances dataset = source.getDataSet();
        
        // Set class index to the last attribute
        dataset.setClassIndex(dataset.numAttributes() - 1);
        
        // Set percentage split for training and testing (80% train, 20% test)
        double trainPercentage = 80.0;

        // Create RemovePercentage filter for splitting data
        RemovePercentage filter = new RemovePercentage();
        filter.setInputFormat(dataset);
        filter.setPercentage(trainPercentage);
        filter.setInvertSelection(true); // Invert selection to keep the remaining instances as training set

        // Apply filter to obtain training set
        Instances trainingSet = Filter.useFilter(dataset, filter);

        // Save training set to ARFF file
        ArffSaver saverTrain = new ArffSaver();
        saverTrain.setInstances(trainingSet);
        saverTrain.setFile(new File("D:\\Year 4\\Data Mining\\Project\\data\\sales_train.arff"));
        saverTrain.writeBatch();

        // Create the inverse of the filter to obtain test set
        RemovePercentage testFilter = new RemovePercentage();
        testFilter.setInputFormat(dataset);
        testFilter.setPercentage(trainPercentage);
        testFilter.setInvertSelection(false); // Invert selection to keep the remaining instances as test set

        // Apply filter to obtain test set
        Instances testSet = Filter.useFilter(dataset, testFilter);

        // Save test set to ARFF file
        ArffSaver saverTest = new ArffSaver();
        saverTest.setInstances(testSet);
        saverTest.setFile(new File("D:\\Year 4\\Data Mining\\Project\\data\\sales_test.arff"));
        saverTest.writeBatch();

        System.out.println("Training set saved as 'sales_train.arff'");
        System.out.println("Test set saved as 'sales_test.arff'");
        
    }
}
