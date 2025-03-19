package util;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;

/**
 * A convenient tool for generating CDF based on PDF
 * @author sdi
 *
 */
public class GnuplotComputeCDF {

	public static void main(String[] args)
	{
		if(args.length!=2)
		{
			System.out.println("Usage: java GnuplotComputeCDF [pdfDir] [cdfDir]");
			System.out.println("Example: java GnuplotComputeCDF gnuplotDistributionOutputDir/lmh/mmse-0-25-384-cpu gnuplotDistributionOutputDir/lmh/mmse-0-25-384-cpu");
			System.exit(0);
		}
		
		String disDir = args[0];
		String cdfDir = args[1];
		System.out.println("disDir="+disDir);
		System.out.println("cdfDir="+cdfDir);
		List<String> disFileList = PVFile.getFiles(disDir);
		Iterator<String> iter = disFileList.iterator();
		while(iter.hasNext())
		{
			String disFile = iter.next();
			if(!disFile.endsWith("dis"))
				continue;
			String disPath = disDir+"/"+disFile;
			List<String> lineList = PVFile.readFile(disPath);
			if(!isNumeric(lineList.get(0).split("\\s")[0]))
				lineList.remove(0); //remove the field line
			double[] data = ConversionHandler.convertStringList2DoubleArray(lineList, 1);
			double unit = Double.parseDouble(lineList.get(1).split("\\s")[0])-Double.parseDouble(lineList.get(0).split("\\s")[0]);
			double firstData = Double.parseDouble(lineList.get(0).split("\\s")[0]);
			List<String> cdfCountList = new ArrayList<String>(); 
			GnuplotDistribution.genCountCDF(data, cdfCountList, firstData, unit);
			String outputFile = cdfDir+"/"+disFile.replace("dis", "cdf");
			PVFile.print2File(cdfCountList, outputFile);
			System.out.println("generate: "+outputFile);
		}
		System.out.println("done.");
	}
	
	public static boolean isNumeric(String str){
	    //Pattern pattern = Pattern.compile("[0-9]+(\\.?)[0-9]*"); 
		Pattern pattern = Pattern.compile("[-+]?(\\d+(\\.\\d*)?|\\.\\d+)([eE][-+]?\\d+)?[dD]?");
	    return pattern.matcher(str).matches();
	 } 
}
