package util;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;


/**
 * to generate probability density function (PDF)
 * @author sdi
 *
 */
public class GnuplotComputeDistribution {

	public static double maxValue = -1;
	
	/**
	 * Wrong : this function is problematic.
	 * @param rawDataDir
	 * @param fieldID
	 * @return
	 */	
	public static List<String>[] aggregateDistribution(String rawDataDir, int[] fieldID, int distributionIntervals)
	{
		List<Double>[] dataList = new ArrayList[fieldID.length];
		for(int i = 0;i<fieldID.length;i++)
			dataList[i] = new ArrayList<Double>();
		List<String> fileList = PVFile.getFiles(rawDataDir);
		Collections.sort(fileList);
		Iterator<String> iter = fileList.iterator();
		long initLogTime = System.currentTimeMillis()/1000;
		
		for(int i = 1;iter.hasNext();i++)
		{
			String fileName = iter.next();
			List<String> lineList = PVFile.readFile(rawDataDir+"/"+fileName);
			Iterator<String> iter2 = lineList.iterator();
			while(iter2.hasNext())
			{
				String line = iter2.next();
				if(line.contains("#")||line.contains("success"))
				{
					continue;
				}
				String[] data = line.split("\\s");
				for(int j = 0;j<fieldID.length;j++)
					dataList[j].add(Double.valueOf(data[fieldID[j]]));
			}
			if(i%10==0)
				PVFile.showProgress(initLogTime, i, fileList.size(), fileName);
		}
		//sorting
		for(int i = 0;i<dataList.length;i++)
		{
			System.out.println("Sorting data .... ("+(i+1)+"/"+dataList.length+")");
			Collections.sort(dataList[i]);
		}
		//convert dataList to data array		
		System.out.println("Converting list to data array ....");
		double[][] rawData = ConversionHandler.convertDoubleLists2DoubleArrays(dataList);
		List<String>[] resultList = new ArrayList[fieldID.length];
		for(int i = 0;i<resultList.length;i++)
			resultList[i] = new ArrayList<String>();
		System.out.println("Generating distribution .....");
		GnuplotDistribution.genDistribution(rawData, resultList, distributionIntervals);
		
		return resultList;
	}
	
	public static List<String>[] aggregateDistribution(String rawDataDir, int[] fieldID, double min, double max, int distributionIntervals)
	{
		List<Double>[] dataList = new ArrayList[fieldID.length];
		for(int i = 0;i<fieldID.length;i++)
			dataList[i] = new ArrayList<Double>();
		List<String> fileList = PVFile.getFiles(rawDataDir);
		Collections.sort(fileList);
		Iterator<String> iter = fileList.iterator();
		long initLogTime = System.currentTimeMillis()/1000;
		
		for(int i = 1;iter.hasNext();i++)
		{
			String fileName = iter.next();
			List<String> lineList = PVFile.readFile(rawDataDir+"/"+fileName);
			Iterator<String> iter2 = lineList.iterator();
			while(iter2.hasNext())
			{
				String line = iter2.next();
				if(line.contains("#")||line.contains("success"))
				{
					continue;
				}
				String[] data = line.split("\\s");
				for(int j = 0;j<fieldID.length;j++)
					dataList[j].add(Double.valueOf(data[fieldID[j]]));
			}
			if(i%10==0)
				PVFile.showProgress(initLogTime, i, fileList.size(), fileName);
		}
		//sorting
		for(int i = 0;i<dataList.length;i++)
		{
			System.out.println("Sorting data .... ("+(i+1)+"/"+dataList.length+")");
			Collections.sort(dataList[i]);
		}
		//convert dataList to data array		
		System.out.println("Converting list to data array ....");
		double[][] rawData = ConversionHandler.convertDoubleLists2DoubleArrays(dataList);
		List<String>[] resultList = new ArrayList[fieldID.length];
		for(int i = 0;i<resultList.length;i++)
			resultList[i] = new ArrayList<String>();
		System.out.println("Generating distribution .....");
		GnuplotDistribution.genDistribution(rawData, resultList, min, max, distributionIntervals);
		
		return resultList;
	}	
	
	private static List<String> initResultList(List<String> inputList, int fieldID)
	{
		List<String> resultList = new ArrayList<String>();
		String[] fields = inputList.get(0).split(" ");
		if(!GnuplotComputeCDF.isNumeric(fields[fieldID]))
			resultList.add(fields[0]+" "+fields[fieldID]);		
		return resultList;
	}

	public static List<String>[] computeDistribution(String rawDataFile, int[] fieldID, int distributionIntervals)
	{	
		System.out.println("Loading raw data from the rawDataFile: "+rawDataFile);
		List<String> inputList = PVFile.readFile(rawDataFile);
		for(int i = 0;i<inputList.size();i++)
		{
			if(inputList.get(i).startsWith("##"))
			{
				inputList.remove(i);
				i--;
			}
			else
				break;
		}
		if(inputList.isEmpty())
		{
			List<String>[] list = new List[fieldID.length];
			for(int i = 0;i<fieldID.length;i++)
				list[i] = new ArrayList<String>();
			return list;
		}
		double[][] rawData = new double[fieldID.length][inputList.size()];
		Iterator<String> it = inputList.iterator();
//		if(!GnuplotComputeCDF.isNumeric(inputList.get(0).split("\\s")[fieldID[0]]))
//			it.next(); //filter out the field line
		for(int j = 0;it.hasNext();j++)
		{
			String s = it.next();
			if(s.startsWith("#"))
				continue;
			String data[] = s.split("\\s");
			if(!GnuplotComputeCDF.isNumeric(data[fieldID[0]]))
				continue;
			for(int i=0;i<fieldID.length;i++)
			{
				rawData[i][j] = Double.parseDouble(data[fieldID[i]]);
			}
		}
		System.out.println("Computing PDf .....");
		List<String>[] resultList = new ArrayList[fieldID.length];
		for(int i = 0;i<fieldID.length;i++)
			resultList[i] = initResultList(inputList, fieldID[i]);
		GnuplotDistribution.genDistribution(rawData, resultList, distributionIntervals);
	
		return resultList;
	}
	
	public static List<String>[] computeDistribution(String rawDataFile, int[] fieldID, double min, double max, int distributionIntervals)
	{	
		System.out.println("Loading raw data from the rawDataFile: "+rawDataFile);
		List<String> inputList = PVFile.readFile(rawDataFile);
		for(int i = 0;i<inputList.size();i++)
		{
			if(inputList.get(i).startsWith("##"))
			{
				inputList.remove(i);
				i--;
			}
			else
				break;
		}
		if(inputList.isEmpty())
		{
			List<String>[] list = new List[fieldID.length];
			for(int i = 0;i<fieldID.length;i++)
				list[i] = new ArrayList<String>();
			return list;
		}
		double[][] rawData = new double[fieldID.length][inputList.size()];
		Iterator<String> it = inputList.iterator();
//		if(!GnuplotComputeCDF.isNumeric(inputList.get(0).split("\\s")[fieldID[0]]))
//			it.next(); //filter out the field line
		for(int j = 0;it.hasNext();j++)
		{
			String s = it.next();
			if(s.startsWith("#"))
				continue;
			String data[] = s.split("\\s");
			if(!GnuplotComputeCDF.isNumeric(data[fieldID[0]]))
				continue;
			for(int i=0;i<fieldID.length;i++)
			{
				rawData[i][j] = Double.parseDouble(data[fieldID[i]]);
			}
		}
		System.out.println("Computing PDf .....");
		List<String>[] resultList = new ArrayList[fieldID.length];
		for(int i = 0;i<fieldID.length;i++)
			resultList[i] = initResultList(inputList, fieldID[i]);
		GnuplotDistribution.genDistribution(rawData, resultList, min, max, distributionIntervals);
	
		return resultList;
	}
	
	public static void generateDistributionGNUPlotFile(String rawDataFile, int fieldIndex, int distributionIntervals, String outputDir)
	{
		int[] fIndex = {fieldIndex};
		 List<String>[] resultList = computeDistribution(rawDataFile, fIndex, distributionIntervals);
		 String[] s = rawDataFile.split("/");
		 String intputDataFileName = s[s.length-1];
		 for(int i = 0;i<fIndex.length;i++)
		 {
			 String outputFile = outputDir+"/"+intputDataFileName+"_"+fIndex[i]+".dis";
			 System.out.println("outputFile="+outputFile);
			 PVFile.print2File(resultList[i], outputFile);
		 }
	}	
	
	public static void generateDistributionGNUPlotFile(String rawDataFile, int fieldIndex, double min, double max, int distributionIntervals, String outputDir)
	{
		int[] fIndex = {fieldIndex};
		 List<String>[] resultList = computeDistribution(rawDataFile, fIndex, min, max, distributionIntervals);
		 String[] s = rawDataFile.split("/");
		 String intputDataFileName = s[s.length-1];
		 for(int i = 0;i<fIndex.length;i++)
		 {
			 String outputFile = outputDir+"/"+intputDataFileName+"_"+fIndex[i]+".dis";
			 System.out.println("outputFile="+outputFile);
			 PVFile.print2File(resultList[i], outputFile);
		 }
	}	
	
	public static void generateDistributionGNUPlotFile_Dir(String rawDataDir, int[] fIndex, int distributionIntervals, 
			String extension, String outputDir)
	{
		List<String> fileList = PVFile.getFiles(rawDataDir);
		Iterator<String> iter = fileList.iterator();
		while(iter.hasNext())
		{
			String fileName = iter.next();
			if(!fileName.endsWith(extension))
				continue;
			String filePath = rawDataDir+"/"+fileName;
			for(int i = 0;i<fIndex.length;i++)
			{
				generateDistributionGNUPlotFile(filePath, fIndex[i], distributionIntervals, outputDir);
			}			
		}
	}
	
	public static void main(String[] args)
	{
		if(args.length<5)
		{
			System.out.println("Usage: java GnuplotComputeDistribution [rawDataFile] [outputDir] [distributionIntervals] [extension] [maxValue] [fieldIndexList]");
			System.out.println("Example: java GnuplotComputeDistribution test.txt dir 100 data 1 2");
			System.out.println("Example: java GnuplotCOmputeDistribution /home/fti/workspace-java/SDCDetection/Nek5kdatatrace_o/quadratic/0/total2.txt /home/fti/workspace-java/SDCDetection/Nek5kdatatrace_o/quadratic/0/pdf 5000000 txt -1 5");
			System.exit(0);
		}
		
		String rawDataPath = args[0];
		String outputDir = args[1];
		int distributionIntervals = Integer.parseInt(args[2]);
		String extension = args[3];
		maxValue = Double.parseDouble(args[4]);
		int[] fieldID = new int[args.length-5];
		for(int i = 0;i<fieldID.length;i++)
			fieldID[i] = Integer.parseInt(args[i+5]);
		
		System.out.println("Plot Distribution for "+rawDataPath);
		System.out.println("Begin....");
//		System.out.println("fieldID[0]="+fieldID[0]);
		
		if(new File(rawDataPath).isDirectory())
		{
			generateDistributionGNUPlotFile_Dir(rawDataPath, fieldID, distributionIntervals, extension, outputDir);
		}
		else
		{
			generateDistributionGNUPlotFile(rawDataPath, fieldID[0], distributionIntervals, outputDir);
		}
		
		System.out.println("Done.");
	}
	
	public static boolean isNumeric(String str){
	    //Pattern pattern = Pattern.compile("[0-9]+(\\.?)[0-9]*"); 
		Pattern pattern = Pattern.compile("[-+]?(\\d+(\\.\\d*)?|\\.\\d+)([eE][-+]?\\d+)?[dD]?");
	    return pattern.matcher(str).matches();
	 } 
}
