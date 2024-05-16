package weka.api;

//import required classes
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;

public class CSV2Arff {
	public static void main(String[] args) throws Exception {
		
		// Load CSV
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File("D:\\Year 4\\Data Mining\\Project\\data\\sales_clothes.csv"));
		Instances data = loader.getDataSet(); //get instances object
		
		// Save ARFF
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data); //set the data we want to convert
		
		// Save as ARFF file
		saver.setFile(new File("D:\\Year 4\\Data Mining\\Project\\data\\sales_clothes.arff"));
		saver.writeBatch();
	}
}
