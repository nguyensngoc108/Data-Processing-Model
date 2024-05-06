package weka.api;

//import required classes
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.core.converters.ArffSaver;
import java.io.File;
import weka.core.converters.ConverterUtils.DataSource;

public class AttrSelection{
	public static void main(String args[]) throws Exception{
		//load dataset
		DataSource source = new DataSource("D:\\Year 4\\Data Mining\\data\\filtered_sales_clothes.arff");
		Instances filteredData = source.getDataSet();
		
        // Set the index of the target attribute
        int targetAttributeIndex = 2; // Change this to the index of your target attribute
        // Set the target attribute
        filteredData.setClassIndex(targetAttributeIndex);

		//create AttributeSelection object
		AttributeSelection filter = new AttributeSelection();
		//create evaluator and search algorithm objects
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		//set the algorithm to search backward
		search.setSearchBackwards(true);
		//set the filter to use the evaluator and search algorithm
		filter.setEvaluator(eval);
		filter.setSearch(search);
		//specify the data
		filter.setInputFormat(filteredData);
		//apply
		Instances newData = Filter.useFilter(filteredData, filter);
		//save
		ArffSaver saver = new ArffSaver();
		saver.setInstances(newData);
		saver.setFile(new File("D:\\Year 4\\Data Mining\\data\\dimension_reduction_sales_clothes.arff"));
		saver.writeBatch();
	}
}