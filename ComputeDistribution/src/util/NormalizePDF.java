package util;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class NormalizePDF {

	public static int size = 2000;
	
	public static void main(String[] args)
	{
		String dir = "/home/sdi/Development/eclipse-workspace/ComputeDistribution";
		String ext = "dis";
		
		List<String> fileList = PVFile.getFiles(dir, ext);
		Iterator<String> iter = fileList.iterator();
		while(iter.hasNext())
		{
			String file = iter.next();
			String filePath = dir+"/"+file;
			String outputFile = filePath+".nor";
			
			float[] data = new float[size];
			int i = 0;
			
			List<String> lineList = PVFile.readFile(filePath);
			Iterator<String> iter2 = lineList.iterator();
			while(iter2.hasNext())
			{
				String line = iter2.next();
				String[] s = line.split("\\s");
				data[i++] = Float.parseFloat(s[1]);
			}
			
			float max = data[0];
			for(i = 0;i<data.length;i++)
			{
				if(data[i]>max)
					max = data[i];
			}
			
			float[] ndata = new float[size];
			for(i=0;i<data.length;i++)
				ndata[i] = data[i]/max;
			
			List<String> resultList = new ArrayList<String>();
			
			i = 0;
			Iterator<String> iter3 = lineList.iterator();
			while(iter3.hasNext())
			{
				String line = iter3.next();
				String[] s = line.split("\\s");
				resultList.add(s[0]+" "+ndata[i++]);
			}
			
			PVFile.print2File(resultList, outputFile);
		}
		
	}
}
