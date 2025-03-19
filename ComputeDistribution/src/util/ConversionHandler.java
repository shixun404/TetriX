package util;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;

public class ConversionHandler {

	public static byte[] convertFloatArray2ByteArray(float[] data)
	{
		byte[] bytes = new byte[data.length*4];
		for(int i = 0;i<data.length;i++)
		{
			int ivalue = Float.floatToIntBits(data[i]);
			byte[] tmp = convertInt2Bytes(ivalue);
			for(int j = 0;j<4;j++)
				bytes[4*i+j]=tmp[j];
		}
		return bytes;
	}
	
	public static float[] convertFloatList2FloatArray(List<Float> list) {
		float[] array = new float[list.size()];
		Iterator<Float> iter = list.iterator();
		for (int i = 0; iter.hasNext(); i++) {
			float v = iter.next();
			array[i] = v;
		}
		return array;
	}

	public static int[] convertIntegerList2IntegerArray(List<Integer> list) {
		int i = 0;
		int[] result = new int[list.size()];
		Iterator<Integer> iter = list.iterator();
		while (iter.hasNext()) {
			int a = iter.next();
			result[i] = a;
			i++;
		}
		return result;
	}

	public static List<Double> convertDoubleArray2DoubleList(double[] data) {
		List<Double> dataList = new ArrayList<Double>();
		for (int i = 0; i < data.length; i++)
			dataList.add(data[i]);
		return dataList;
	}

	public static double[] convertDoubleList2DoubleArray(List<Double> list) {
		double[] array = new double[list.size()];
		Iterator<Double> iter = list.iterator();
		for (int i = 0; iter.hasNext(); i++) {
			double v = iter.next();
			array[i] = v;
		}
		return array;
	}

	public static List<Double> convertStringList2DoubleList(List<String> list) {
		List<Double> valueList = new ArrayList<Double>();
		Iterator<String> iter = list.iterator();
		while (iter.hasNext()) {
			String s = iter.next().trim();
			if(s.startsWith("#"))
				continue;
			double value = Double.parseDouble(s);
			valueList.add(value);
		}
		return valueList;
	}

	public static double[] convertArray2List4Comp(double[] countVector,
			List<String> countList, double offset, double x_unit) {
		double[] result = new double[countVector.length];

		int size = countVector.length;

		double sum = 0;
		for (int i = 0; i < size; i++)
			sum += countVector[i];

		for (int i = 0; i < size; i++) {
			String s = String.valueOf(offset + x_unit * (i + 1));
			result[i] = countVector[i] / sum;
			s += " " + String.valueOf(result[i]);
			countList.add(s);
		}
		return result;
	}

	public static List<String> convertDoubleArray2StringListWithIndexes(
			double[] data) {
		List<String> lineList = new ArrayList<String>();
		for (int i = 0; i < data.length; i++)
			lineList.add(i + " " + String.valueOf(data[i]));
		return lineList;
	}
	
	public static List<String> convertFloatArray2StringListWithIndexes(
			float[] data) {
		List<String> lineList = new ArrayList<String>();
		for (int i = 0; i < data.length; i++)
			lineList.add(i + " " + String.valueOf(data[i]));
		return lineList;
	}

	public static float[] convertDoubleList2FloatArray(List<Double> dataList) {
		float[] newData = new float[dataList.size()];
		Iterator<Double> iter = dataList.iterator();
		for (int i = 0; iter.hasNext(); i++) {
			float data = iter.next().floatValue();
			newData[i] = data;
		}
		return newData;
	}

	public static List<String> convertFloatArray2StringList(float[] data) {
		List<String> lineList = new ArrayList<String>();
		for (int i = 0; i < data.length; i++)
			lineList.add(String.valueOf(data[i]));
		return lineList;
	}
	
	public static List<String> convertDoubleArray2StringList(double[] data) {
		List<String> lineList = new ArrayList<String>();
		for (int i = 0; i < data.length; i++)
			lineList.add(String.valueOf(data[i]));
		return lineList;
	}

	public static double[][] convertDoubleLists2DoubleArrays(
			List<Double>[] dataList) {
		int size = dataList[0].size();
		double[][] dataArrays = new double[dataList.length][size];
		for (int i = 0; i < dataList.length; i++) {
			Iterator<Double> iter = dataList[i].iterator();
			for (int j = 0; iter.hasNext(); j++) {
				dataArrays[i][j] = iter.next();
			}
		}
		return dataArrays;
	}

	public static float[] convertStringList2FloatArray(List<String> dataList, int index)
	{
		List<Float> resultList = new ArrayList<Float>();
		Iterator<String> iter = dataList.iterator();
		while(iter.hasNext())
		{
			String s = iter.next();
			if(s.startsWith("#"))
				continue;
			String[] data = s.trim().split("\\s");
			if(isNumeric(data[index]))
				resultList.add(Float.valueOf(data[index]));
		}
		float[] result = new float[resultList.size()];
		Iterator<Float> iter2 = resultList.iterator();
		for (int i = 0; iter2.hasNext(); i++)
			result[i] = iter2.next();
		return result;
	}
	
	public static List<Float> convertStringList2FloatList(List<String> dataList, int index)
	{
		List<Float> resultList = new ArrayList<Float>();
		Iterator<String> iter = dataList.iterator();
		while(iter.hasNext())
		{
			String s = iter.next();
			if(s.startsWith("#"))
				continue;
			String[] data = s.trim().split("\\s");
			if(isNumeric(data[index]))
				resultList.add(Float.valueOf(data[index]));
		}
		return resultList;
	}
	
	public static double[] convertStringList2DoubleArray(List<String> dataList,
			int index) {
		List<Double> resultList = new ArrayList<Double>();
		Iterator<String> iter = dataList.iterator();
		while (iter.hasNext()) {
			String s = iter.next();
			if (s.startsWith("#"))
				continue;
			String[] data = s.trim().split("\\s");
			if (isNumeric(data[index]))
				resultList.add(Double.valueOf(data[index]));
		}
		double[] result = new double[resultList.size()];
		Iterator<Double> iter2 = resultList.iterator();
		for (int i = 0; iter2.hasNext(); i++)
			result[i] = iter2.next();
		return result;
	}

	public static double[] copy(double[] data) {
		double[] newData = new double[data.length];
		for (int i = 0; i < newData.length; i++)
			newData[i] = data[i];
		return newData;
	}

	public static boolean isNumeric(String str) {
		// Pattern pattern = Pattern.compile("[0-9]+(\\.?)[0-9]*");
		Pattern pattern = Pattern
				.compile("[-+]?(\\d+(\\.\\d*)?|\\.\\d+)([eE][-+]?\\d+)?[dD]?");
		return pattern.matcher(str).matches();
	}

	public static double[] mergeDoubleArray(double[] a1, double[] a2) {
		double[] newA = new double[a1.length + a2.length];
		for (int i = 0; i < a1.length; i++)
			newA[i] = a1[i];
		for (int i = 0, j = a1.length; i < a2.length; i++, j++)
			newA[j] = a2[i];

		return newA;
	}

	public static int[] convertIntList2Array(List<Integer> intList) {
		int[] array = new int[intList.size()];
		Iterator<Integer> iter = intList.iterator();
		for (int i = 0; iter.hasNext(); i++) {
			int v = iter.next();
			array[i] = v;
		}
		return array;
	}
	
	public static List<Integer> convertIntArray2IntList(int[] data)
	{
		List<Integer> lineList = new ArrayList<Integer>();
		for(int i = 0;i<data.length;i++)
			lineList.add(new Integer(data[i]));
		return lineList;
	}

	public static Object convertOriData2MultiArray(float[] data, int[] dimSize) {
		switch (dimSize.length) {
		case 1:
			return data;
		case 2:
			float[][] newData2 = new float[dimSize[1]][dimSize[0]];
			for (int i = 0; i < dimSize[1]; i++)
				for (int j = 0; j < dimSize[0]; j++)
					newData2[i][j] = getOriData(data, dimSize,
							new int[] { j, i });
			return newData2;
		case 3:
			float[][][] newData3 = new float[dimSize[2]][dimSize[1]][dimSize[0]];
			for (int i = 0; i < dimSize[2]; i++)
				for (int j = 0; j < dimSize[1]; j++)
					for (int k = 0; k < dimSize[0]; k++)
						newData3[i][j][k] = getOriData(data, dimSize,
								new int[] { k, j, i });
			return newData3;
		case 4:
			float[][][][] newData4 = new float[dimSize[3]][dimSize[2]][dimSize[1]][dimSize[0]];
			for (int i = 0; i < dimSize[3]; i++)
				for (int j = 0; j < dimSize[2]; j++)
					for (int k = 0; k < dimSize[1]; k++)
						for (int p = 0; p < dimSize[0]; p++)
							newData4[i][j][k][p] = getOriData(data, dimSize,
									new int[] { p, k, j, i });
			return newData4;
		case 5:
			float[][][][][] newData5 = new float[dimSize[4]][dimSize[3]][dimSize[2]][dimSize[1]][dimSize[0]];
			for (int i = 0; i < dimSize[4]; i++)
				for (int j = 0; j < dimSize[3]; j++)
					for (int k = 0; k < dimSize[2]; k++)
						for (int p = 0; p < dimSize[1]; p++)
							for (int q = 0; q < dimSize[0]; q++)
								newData5[i][j][k][p][q] = getOriData(data,
										dimSize, new int[] { q, p, k, j, i });
			return newData5;
		case 6:
			float[][][][][][] newData6 = new float[dimSize[5]][dimSize[4]][dimSize[3]][dimSize[2]][dimSize[1]][dimSize[0]];
			for (int i = 0; i < dimSize[5]; i++)
				for (int j = 0; j < dimSize[4]; j++)
					for (int k = 0; k < dimSize[3]; k++)
						for (int p = 0; p < dimSize[2]; p++)
							for (int q = 0; q < dimSize[1]; q++)
								for (int r = 0; r < dimSize[0]; r++)
									newData6[i][j][k][p][q][r] = getOriData(
											data, dimSize, new int[] { r, q, p,
													k, j, i });
			return newData6;
		}
		return null;
	}
	
	public static Object convertOriData2MultiArray(double[] data, int[] dimSize) {
		switch (dimSize.length) {
		case 1:
			return data;
		case 2:
			double[][] newData2 = new double[dimSize[1]][dimSize[0]];
			for (int i = 0; i < dimSize[1]; i++)
				for (int j = 0; j < dimSize[0]; j++)
					newData2[i][j] = getOriData(data, dimSize,
							new int[] { j, i });
			return newData2;
		case 3:
			double[][][] newData3 = new double[dimSize[2]][dimSize[1]][dimSize[0]];
			for (int i = 0; i < dimSize[2]; i++)
				for (int j = 0; j < dimSize[1]; j++)
					for (int k = 0; k < dimSize[0]; k++)
						newData3[i][j][k] = getOriData(data, dimSize,
								new int[] { k, j, i });
			return newData3;
		case 4:
			double[][][][] newData4 = new double[dimSize[3]][dimSize[2]][dimSize[1]][dimSize[0]];
			for (int i = 0; i < dimSize[3]; i++)
				for (int j = 0; j < dimSize[2]; j++)
					for (int k = 0; k < dimSize[1]; k++)
						for (int p = 0; p < dimSize[0]; p++)
							newData4[i][j][k][p] = getOriData(data, dimSize,
									new int[] { p, k, j, i });
			return newData4;
		case 5:
			double[][][][][] newData5 = new double[dimSize[4]][dimSize[3]][dimSize[2]][dimSize[1]][dimSize[0]];
			for (int i = 0; i < dimSize[4]; i++)
				for (int j = 0; j < dimSize[3]; j++)
					for (int k = 0; k < dimSize[2]; k++)
						for (int p = 0; p < dimSize[1]; p++)
							for (int q = 0; q < dimSize[0]; q++)
								newData5[i][j][k][p][q] = getOriData(data,
										dimSize, new int[] { q, p, k, j, i });
			return newData5;
		case 6:
			double[][][][][][] newData6 = new double[dimSize[5]][dimSize[4]][dimSize[3]][dimSize[2]][dimSize[1]][dimSize[0]];
			for (int i = 0; i < dimSize[5]; i++)
				for (int j = 0; j < dimSize[4]; j++)
					for (int k = 0; k < dimSize[3]; k++)
						for (int p = 0; p < dimSize[2]; p++)
							for (int q = 0; q < dimSize[1]; q++)
								for (int r = 0; r < dimSize[0]; r++)
									newData6[i][j][k][p][q][r] = getOriData(
											data, dimSize, new int[] { r, q, p,
													k, j, i });
			return newData6;
		}
		return null;
	}

	public static double getOriData(double[] data, int[] dimSize, int[] index) {
		int loc = 0;
		for (int i = 0; i <= dimSize.length - 1; i++) {
			int addEle = index[i];
			for (int j = 0; j < i; j++)
				addEle *= dimSize[j];
			loc += addEle;
		}
		return data[loc];
	}
	
	public static float getOriData(float[] data, int[] dimSize, int[] index) {
		int loc = 0;
		for (int i = 0; i <= dimSize.length - 1; i++) {
			int addEle = index[i];
			for (int j = 0; j < i; j++)
				addEle *= dimSize[j];
			loc += addEle;
		}
		return data[loc];
	}

	public static String convertDouble2ByteString(double value) {
		StringBuffer sb = new StringBuffer();
		long long_value = Double.doubleToLongBits(value);
		String sh = Long.toBinaryString(long_value);
		char[] bits = sh.toCharArray();
		if (bits.length < 64) {
			sb.append("0|");
			for (int i = 0; i < 11; i++)
				sb.append(bits[i]);
			sb.append("|");
			for (int i = 11; i < bits.length; i++)
				sb.append(bits[i]);
		} else {
			sb.append(bits[0] + "|");
			for (int i = 1; i < 12; i++)
				sb.append(bits[i]);
			sb.append("|");
			for (int i = 12; i < bits.length; i++)
				sb.append(bits[i]);
		}
		return sb.toString();
	}

	public static BitSet convertDouble2BitSet(double value) {
		long long_value = Double.doubleToLongBits(value);
		BitSet bs = BitSet.valueOf(new long[] { long_value });
		return bs;
	}
	
	public static BitSet convertFloat2BitSet(float value) {
		long long_value = Float.floatToIntBits(value);
		BitSet bs = BitSet.valueOf(new long[] { long_value });
		return bs;
	}

	public static BitSet convertDouble2BitSet(double[] values) {
		long[] long_values = new long[values.length];
		for (int i = 0; i < values.length; i++)
			long_values[i] = Double.doubleToLongBits(values[i]);
		BitSet bs = BitSet.valueOf(long_values);
		return bs;
	}
	
	public static BitSet convertFloat2BitSet(float[] values) {
		long[] long_values;
		if(values.length%2==0)
		{
			long_values = new long[values.length/2];
			long prevLong_value = 0;
			for (int i = 0; i < values.length; i++)
			{
				if(i%2==0)
					prevLong_value = Float.floatToIntBits(values[i]);
				else if(i%2==1)
				{
					long postLong_value = Float.floatToIntBits(values[i]);
					prevLong_value = prevLong_value << 32;
					postLong_value = (postLong_value << 32) >>>32;
					long finalLong_value = prevLong_value | postLong_value;
					long_values[i/2] = finalLong_value;
				}
			}
		}
		else
		{	
			long_values = new long[values.length/2+1];
			long prevLong_value = 0;
			for (int i = 0; i < values.length; i++)
			{
				if(i%2==0)
					prevLong_value = Float.floatToIntBits(values[i]);
				else if(i%2==1)
				{
					long postLong_value = Float.floatToIntBits(values[i]);
					postLong_value = postLong_value << 32;
					long finalLong_value = prevLong_value | postLong_value;
					long_values[i/2] = finalLong_value;
				}
				if(i==values.length-1)
					long_values[i/2] = prevLong_value;
			}
		}

		BitSet bs = BitSet.valueOf(long_values);
		return bs;
	}

	public static double convertBitSet2Double(BitSet bitset) {
		long lvalue = 0;
		if (!bitset.isEmpty())
			lvalue = bitset.toLongArray()[0];
		return Double.longBitsToDouble(lvalue);
	}

	public static float convertBitSet2Float(BitSet bitset) {
		int ivalue = 0;
		if (!bitset.isEmpty())
			ivalue = (int)bitset.toLongArray()[0];
		return Float.intBitsToFloat(ivalue);
	}
	
	public static BitSet convertBitMapArray2BitSet(int[] bitmap) {
		BitSet bs = new BitSet(bitmap.length);
		for (int i = 0; i < bitmap.length; i++) {
			if (bitmap[i] == 1)
				bs.set(i);
		}
		return bs;
	}

	public static double[] convertMultiArrayTo1DArray(double[][] data) {
		double[] result = new double[data.length * data[0].length];
		int m = 0;
		for (int i = 0; i < data.length; i++)
			for (int j = 0; j < data[0].length; j++)
				result[m++] = data[i][j];
		return result;
	}
	
	public static float[] convertMultiArrayTo1DArray(float[][] data) {
		float[] result = new float[data.length * data[0].length];
		int m = 0;
		for (int i = 0; i < data.length; i++)
			for (int j = 0; j < data[0].length; j++)
				result[m++] = data[i][j];
		return result;
	}

	public static double[] convertMultiArrayTo1DArray(double[][][] data) {
		double[] result = new double[data.length * data[0].length
				* data[0][0].length];
		int m = 0;
		for (int i = 0; i < data.length; i++)
			for (int j = 0; j < data[0].length; j++)
				for (int k = 0; k < data[0][0].length; k++)
					result[m++] = data[i][j][k];
		return result;
	}
	
	public static float[] convertMultiArrayTo1DArray(float[][][] data) {
		float[] result = new float[data.length * data[0].length
				* data[0][0].length];
		int m = 0;
		for (int i = 0; i < data.length; i++)
			for (int j = 0; j < data[0].length; j++)
				for (int k = 0; k < data[0][0].length; k++)
					result[m++] = data[i][j][k];
		return result;
	}

	public static double[] convertMultiArrayTo1DArray(double[][][][] data) {
		double[] result = new double[data.length * data[0].length
				* data[0][0].length * data[0][0][0].length];
		int m = 0;
		for (int i = 0; i < data.length; i++)
			for (int j = 0; j < data[0].length; j++)
				for (int k = 0; k < data[0][0].length; k++)
					for (int z = 0; z < data[0][0][0].length; z++)
						result[m++] = data[i][j][k][z];
		return result;
	}
	
	public static float[] convertMultiArrayTo1DArray(float[][][][] data) {
		float[] result = new float[data.length * data[0].length
				* data[0][0].length * data[0][0][0].length];
		int m = 0;
		for (int i = 0; i < data.length; i++)
			for (int j = 0; j < data[0].length; j++)
				for (int k = 0; k < data[0][0].length; k++)
					for (int z = 0; z < data[0][0][0].length; z++)
						result[m++] = data[i][j][k][z];
		return result;
	}

	public static double[] convertMultiArrayTo1DArray(double[][][][][] data) {
		double[] result = new double[data.length * data[0].length
				* data[0][0].length * data[0][0][0].length
				* data[0][0][0][0].length];
		int m = 0;
		for (int i = 0; i < data.length; i++)
			for (int j = 0; j < data[0].length; j++)
				for (int k = 0; k < data[0][0].length; k++)
					for (int z = 0; z < data[0][0][0].length; z++)
						for (int w = 0; w < data[0][0][0][0].length; w++)
							result[m++] = data[i][j][k][z][w];
		return result;
	}
	
	public static float[] convertMultiArrayTo1DArray(float[][][][][] data) {
		float[] result = new float[data.length * data[0].length
				* data[0][0].length * data[0][0][0].length
				* data[0][0][0][0].length];
		int m = 0;
		for (int i = 0; i < data.length; i++)
			for (int j = 0; j < data[0].length; j++)
				for (int k = 0; k < data[0][0].length; k++)
					for (int z = 0; z < data[0][0][0].length; z++)
						for (int w = 0; w < data[0][0][0][0].length; w++)
							result[m++] = data[i][j][k][z][w];
		return result;
	}

	public static double[] convertByteArray2DoubleArray(byte[] data) {
		if (data.length % 8 != 0) {
			System.out
					.println("Error: The input data length is not multiple of 8 bytes!\n It may not be floating-point data array.");
			System.exit(0);
		}
		int length = data.length/8;
		BitSet bs = BitSet.valueOf(data);
		long[] longData = bs.toLongArray();
		double[] doubleData = new double[length];
		for (int i = 0; i < longData.length; i++) {
			doubleData[i] = Double.longBitsToDouble(longData[i]); //TODO: This step may be incorrect!
		}
		if(longData.length<length)
		{
			for(int i = longData.length;i<length;i++)
				doubleData[i] = 0;
		}
		return doubleData;
	}
	
	public static float[] convertByteArray2FloatArray(byte[] bytes) {
		if (bytes.length % 4 != 0) {
			System.out
					.println("Error: The input data length is not multiple of 8 bytes!\n It may not be floating-point data array.");
			System.exit(0);
		}
		int length = bytes.length/4;
		float[] floatData = new float[length];
		for(int i = 0;i<floatData.length;i++)
		{
			int tmp = ((int)bytes[i*4+3]) << 24;
			tmp =  tmp | (bytes[i*4+2] << 24) >>> 8;
			tmp = tmp | (bytes[i*4+1] << 24) >>> 16;
			tmp = tmp | (bytes[i*4+0] << 24) >>> 24;
			float vv = Float.intBitsToFloat(tmp);
			floatData[i] = vv;
		}
		
		return floatData;
	}
	
	public static byte[] convertDoubleArray2ByteArray(double[] data)
	{
		byte[] bytes = new byte[data.length*8];
		long[] longData = new long[data.length];
	
		for(int i = 0;i<data.length;i++)
			longData[i] = Double.doubleToLongBits(data[i]);
		
		BitSet bs = BitSet.valueOf(longData);
		byte[] tmpBytes = bs.toByteArray();
		
		int i = 0;
		for(;i<tmpBytes.length;i++)
			bytes[i] = tmpBytes[i];
		if(i<bytes.length)
			for(;i<bytes.length;i++)
				bytes[i] = 0;
		
		return bytes;
	}
	
	public static byte[] convertDoubleArray2ByteArray(float[] data)
	{
		int[] intData = new int[data.length];
	
		for(int i = 0;i<data.length;i++)
			intData[i] = Float.floatToIntBits(data[i]);
		
		byte[] bytes = ConversionHandler.convertIntArray2Bytes(intData);
		
		return bytes;
	}
	

	public static byte[] convertObject2Bytes(Object object) {
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		ObjectOutput out = null;
		try {
			out = new ObjectOutputStream(bos);
			out.writeObject(object);
		} catch (Exception e) {
			e.printStackTrace();
		}
		byte[] newBytes = bos.toByteArray();
		return newBytes;
	}

	public static Object convertByts2Object(byte[] byteData) {
		ByteArrayInputStream bis = new ByteArrayInputStream(byteData);
		ObjectInput in = null;
		try {
			in = new ObjectInputStream(bis);
			Object o = in.readObject();
			return o;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public static byte[] convertInt2Bytes(int value)
	{
		  byte[] result = new byte[4];

		  result[3] = (byte) (value >> 24);
		  result[2] = (byte) (value >> 16);
		  result[1] = (byte) (value >> 8);
		  result[0] = (byte) (value /*>> 0*/);

		  return result;
	}
	
	public static byte[] convertIntArray2Bytes(int[] data)
	{
		byte[] result = new byte[4*data.length];
		
		for(int i= 0;i<data.length;i++)
		{
			byte[] tmp = convertInt2Bytes(data[i]);
			for(int j = 0;j<4;j++)
			{
				result[i*4+j] = tmp[j];
			}
		}
		return result;
	}
	
	public static byte[] convertByteList2ByteArray(List<byte[]> list)
	{
		int size = 0;
		Iterator<byte[]> iter = list.iterator();
		while(iter.hasNext())
		{
			byte[] bb = iter.next();
			size += bb.length;
		}
		
		byte[] result = new byte[size];
		iter = list.iterator();
		int i = 0;
		while(iter.hasNext())
		{
			byte[] bb = iter.next();
			for(int j = 0;j<bb.length;j++)
				result[i++] = bb[j];
		}
		
		return result;
	}
	
	public static void main(String[] args)
	{
		double[] test = new double[2];
		for(int i= 0;i<2;i++)
			test[i] = i;
		byte[] b = convertObject2Bytes(test);
		double[] testnew = (double[])convertByts2Object(b);
		System.out.println("testnew[0]="+testnew[0]+",testnew[1]="+testnew[1]);
	}
}
