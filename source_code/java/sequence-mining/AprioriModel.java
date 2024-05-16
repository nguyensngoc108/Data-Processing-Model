package sequence.mining;

import weka.associations.Apriori;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class AprioriModel {
	public static void main(String args[]) throws Exception{
		//load data
		String dataset = "D:\\Year 4\\Data Mining\\Project\\data\\filtered_basket_sets.arff";
		DataSource source = new DataSource(dataset);
		Instances data = source.getDataSet();
		
		//the Apriori algorithm
		Apriori model = new Apriori();
        String[] options = {"-N", "10", "-T", "1", "-C", "0.9", "-D", "0.05","-M", "0.1", "-V",};
        model.setOptions(options);
        
		//build model
		model.buildAssociations(data);
		System.out.println(model);
		
        // Save Apriori model
//        SerializationHelper.write("D:\\Year 4\\Data Mining\\Project\\models\\sequence mining\\Apriori.model", model);
	}
}
