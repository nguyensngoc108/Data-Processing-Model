package weka.api;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import java.io.File;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class AttributeFilter {
    public static void main(String args[]) throws Exception {
        // Load data
        DataSource source = new DataSource("D:\\Year 4\\Data Mining\\Project\\data\\sales_clothes.arff");
        Instances dataset = source.getDataSet();
        
        // Specify the attributes to convert
        int[] attributesToConvert = {3, 4, 13, 14, 15, 19}; // Change the attribute indices as needed
        // Apply the NumericToNominal filter only to specified attributes
        NumericToNominal numericToNominalFilter = new NumericToNominal();
        numericToNominalFilter.setAttributeIndicesArray(attributesToConvert);
        numericToNominalFilter.setInputFormat(dataset);
        Instances convertedData = Filter.useFilter(dataset, numericToNominalFilter);

        // Remove unnecessary attributes
        String[] opts = new String[]{"-R", "1, 21, 22, 23"};
        // Create a Remove object (this is the filter class)
        Remove remove = new Remove();
        // Set filter options
        remove.setOptions(opts);
        // Pass the data to apply filter
        remove.setInputFormat(convertedData);
        Instances filteredData = Filter.useFilter(convertedData, remove);

        // Now save the data to a new file
        ArffSaver saver = new ArffSaver();
        saver.setInstances(filteredData);
        saver.setFile(new File("D:\\Year 4\\Data Mining\\Project\\data\\filtered_sales_clothes.arff"));
        saver.writeBatch();
    }
}
