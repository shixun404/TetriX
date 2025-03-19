package util;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class GnuplotComputeCDF_VBREL {

	public static void main(String[] args)
	{
		//ARAMCO: 1000:0.112477, 1500:0.082279, 2000:0.068970 2500:0.053396 3000:0.000119
		//String fileName = "/home/sdi/Data/QMCPack/cdf/einspline_288_115_69_69.pre_8.amp_0.cdf";
		String fileName = "/home/sdi/Data/ARAMCO/cdf/aramco-snapshot-3000.f32_128.amp_0.cdf";
		//float valueRange = 0.000119f; //ARAMCO
		//float valueRange = 143.032700f; //Hurricane Uf23
		//float valueRange = 4.414280f; //Miranda
		//float valueRange = 3.321542f; //Nyx temperature
		//float valueRange = 33.010017f; //QMCPACK
		float valueRange = 0.000119f;
		List<String> lineList = PVFile.readFile(fileName);
		Iterator<String> iter = lineList.iterator();
		
		List<String> resultList = new ArrayList<String>();
		while(iter.hasNext())
		{
			String line = iter.next();
			String[] data = line.split("\\s");
			float key =Float.parseFloat(data[0]); //percentage
			float value = Float.parseFloat(data[1]);
			resultList.add(key/valueRange+" "+value*100);
		}
		
		PVFile.print2File(resultList, fileName+"2");
		System.out.println("Output result in "+fileName+"2");
	}
}
