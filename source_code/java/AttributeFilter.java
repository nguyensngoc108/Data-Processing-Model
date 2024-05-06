package weka.api;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import java.io.File;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.RenameAttribute;

public class AttributeFilter {
    public static void main(String args[]) throws Exception {
        // Load data
        DataSource source = new DataSource("D:\\Year 4\\Data Mining\\data\\sales_clothes.arff");
        Instances dataset = source.getDataSet();
        
        // Create a RenameAttribute filter
        String[] str = new String[]{"-R", "first", "-replace", "product_id"};
        // Create a Remove object (this is the filter class)
        RenameAttribute rename = new RenameAttribute();
        // Set filter options
        rename.setOptions(str);
        // Pass the data to apply filter
        rename.setInputFormat(dataset);
        Instances renamedData = Filter.useFilter(dataset, rename);
        

        // Specify the attributes to convert
        int[] attributesToConvert = {3, 4, 13, 14, 15, 19}; // Change the attribute indices as needed
        // Apply the NumericToNominal filter only to specified attributes
        NumericToNominal numericToNominalFilter = new NumericToNominal();
        numericToNominalFilter.setAttributeIndicesArray(attributesToConvert);
        numericToNominalFilter.setInputFormat(renamedData);
        Instances convertedData = Filter.useFilter(renamedData, numericToNominalFilter);

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
        saver.setFile(new File("D:\\Year 4\\Data Mining\\data\\filtered_sales_clothes.arff"));
        saver.writeBatch();
    }
}
