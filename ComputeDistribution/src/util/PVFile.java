package util;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Pattern;
import java.util.zip.Deflater;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class PVFile {
	public static byte[] versionNumber = new byte[]{0,5,14};
	public static int number = 0;
	public static HashMap<String, String> configMap = new HashMap<String, String>();
	
	public static int maxSegmentNum = 1024; //2, or 1024? 
	public static byte OFFSET = 2; //or 
	
	public static int ABS = 0;
	public static int REL = 1;
	public static int ABS_AND_REL = 2;
	public static int ABS_OR_REL = 3;
	
	public static int errBoundMode = ABS;
	
	public static double absBoundRatio = 0.000001;
	public static double relBoundRatio = 1.0E-10;//0.00078125;
	
	public static double compressTime = 0;
	public static double zipTime = 0;
	public static double spacefillingTime = 0;
	public static double computeValueRangeTime = 0;
	public static double writeTime = 0;
	
	
	public static double decompressTime = 0;
	public static double unzipTime = 0; //in seconds
	public static double readTime = 0;
	
	public static DecimalFormat df1 = new DecimalFormat("0000");
	public static DecimalFormat df2 = new DecimalFormat("00000");

	public PVFile() {
	}

	public static boolean isNumeric(String str) {
		// Pattern pattern = Pattern.compile("[0-9]+(\\.?)[0-9]*");
		Pattern pattern = Pattern
				.compile("[-+]?(\\d+(\\.\\d*)?|\\.\\d+)([eE][-+]?\\d+)?[dD]?");
		return pattern.matcher(str).matches();
	}
	
	public static List<String> getSubFile(String path) {
		List<String> list = new LinkedList<String>();
		List<String> ls = getDir(path);
		if (ls.size() > 0) {
			Iterator<String> it = ls.iterator();
			while (it.hasNext()) {
				String curpath = path + File.separator + it.next();
				List<String> sublist = getSubFile(curpath);
				list.addAll(sublist);
			}
		} else {
			ls = getFiles(path);
			if (ls.size() > 0) {
				Iterator<String> it = ls.iterator();
				while (it.hasNext()) {
					String fileName = path + File.separator + it.next();
					list.add(fileName);
				}
			}
		}
		return list;
	}

	public static List<String> getDir(String path) {
		List<String> list = getFileByType(path, 1);
		return list;
	}

	public static List<String> getFiles(String path) {
		List<String> list = getFileByType(path, 2);
		return list;
	}

	public static List<String> getFiles(String path, String extension) {
		List<String> newlist = new ArrayList<String>();
		List<String> list = getFiles(path);
		Iterator<String> it = list.iterator();
		while (it.hasNext()) {
			String fileName = it.next();
			if (fileName.endsWith(extension))
				newlist.add(fileName);
		}
		return newlist;
	}
	
	public static List<String> getFiles(String path, String extension1, String extension2)
	{
		List<String> newlist = new ArrayList<String>();
		List<String> list = getFiles(path);
		Iterator<String> it = list.iterator();
		while (it.hasNext()) {
			String fileName = it.next();
			if (fileName.endsWith(extension1)||fileName.endsWith(extension2))
				newlist.add(fileName);
		}
		return newlist;
	}

	private static List<String> getFileByType(String path, int type) {
		List<String> list = new LinkedList<String>();
		List<String> ls = getLs(path);
		if (ls.size() > 0) {
			Iterator<String> it = ls.iterator();
			while (it.hasNext()) {
				String filename = it.next();
				File f = new File(path, filename);
				if (type == 1 && f.isDirectory())
					list.add(filename);
				else if (type == 2 && f.isFile())
					list.add(filename);
			}
		}
		return list;
	}

	private static List<String> getLs(String path) {
		List<String> list = new LinkedList<String>();
		File dir = new File(path);
		if (dir.isDirectory()) {
			String[] fileNames = dir.list();
			for (int i = 0; i < fileNames.length; i++) {
				list.add(fileNames[i]);
			}
		}
		return list;
	}

	/**
	 * Write a sentence to a file
	 * 
	 * @param filename
	 * @param line
	 * @throws IOException
	 */
	public static void writeRec(String filename, String line)
			throws IOException {
		File output = new File(filename);
		if (!output.exists()) {
			output.createNewFile();
		}
		RandomAccessFile raf = new RandomAccessFile(output, "rw");
		FileChannel fc = raf.getChannel();
		FileLock fl = fc.tryLock();

		if (fl.isValid()) {
			raf.seek(0);
			raf.writeBytes(String.valueOf(line));
			fl.release();
		}
		raf.close();
	}

	/**
	 * Read file's content
	 * 
	 * @param filename
	 * @return
	 */
	public static String read(String filename) {
		String line = "";
		try {
			File input = new File(filename);
			if (!input.exists()) {
				input.createNewFile();
			}
			InputStreamReader read = new InputStreamReader(new FileInputStream(
					input), "GBK");
			BufferedReader reader = new BufferedReader(read);
			line = reader.readLine();
			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return line;
	}

	public static List<String> parseFile(String file) {
		List<String> list = new LinkedList<String>();
		try {
			File input = new File(file);
			InputStreamReader read = new InputStreamReader(new FileInputStream(
					input), "utf-8");
			BufferedReader reader = new BufferedReader(read);
			String line = null;
			while ((line = reader.readLine()) != null) {
				list.add(line);
			}
			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return list;
	}

	public static void createDir(String dirPath) {
		File f = new File(dirPath);
		if (!f.exists())
			f.mkdirs();
	}

	public static boolean deleteFile(String fileName) {
		File file = new File(fileName);
		if (file.isFile() && file.exists()) {
			file.delete();
			System.out.println("Deleting File: " + fileName + " OK delete");
			return true;
		} else {
			System.out.println("Deleting File: " + fileName + " Failed delete");
			return false;
		}
	}

	public static boolean deleteDir(String dir) {

		if (!dir.endsWith(File.separator)) {
			dir = dir + File.separator;
		}
		File dirFile = new File(dir);

		if (!dirFile.exists() || !dirFile.isDirectory()) {
			System.out.println("delete dir failure: " + dir + "");
			return false;
		}
		boolean flag = true;

		File[] files = dirFile.listFiles();
		for (int i = 0; i < files.length; i++) {
			if (files[i].isFile()) {
				flag = deleteFile(files[i].getAbsolutePath());
				if (!flag) {
					break;
				}
			}

			else {
				flag = deleteDir(files[i].getAbsolutePath());
				if (!flag) {
					break;
				}
			}
		}

		if (!flag) {
			System.out.println("failed of deleting file!");
			return false;
		}

		if (dirFile.delete()) {
			System.out.println("deleting " + dir + " OK delte");
			return true;
		} else {
			System.out.println("deleting " + dir + " failed delete");
			return false;
		}
	}

	public static void checkCreateDir(String dirPath) {
		File dir = new File(dirPath);
		if (!dir.exists())
			dir.mkdirs();
	}

	public static boolean isExist(String path) {
		File file = new File(path);
		if (file.exists())
			return true;
		else
			return false;
	}

	public static File checkCreateFile(String filePath) {
		String[] s = filePath.split("/|\\\\");
		File file;
		String dirPath = "";
		for (int i = 0; i < s.length - 1; i++) {
			dirPath += s[i] + "/";
		}
		File dir = new File(dirPath);
		if (!dir.exists())
			dir.mkdirs();
		file = new File(filePath);
		try {
			if (!file.exists())
				file.createNewFile();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return file;
	}

	public static void delete(String filePath) {
		File file = new File(filePath);
		if (file.exists())
			if (!file.delete()) {
				System.err.println("Failure to delete " + file.getPath());
				System.exit(0);
			}
	}

	/**
	 * 
	 * @param srcDirPath
	 * @param extension
	 * @return
	 */
	public static List<String> getRecursiveFiles(String srcDirPath,
			String extension) {
		List<String> fileList = new ArrayList<String>();
		List<String> fileNameList = PVFile.getFiles(srcDirPath);
		Iterator<String> iter = fileNameList.iterator();
		for (; iter.hasNext(); number++) {
			String fileName = iter.next();
			if (fileName.endsWith(extension))
				fileList.add(srcDirPath + "/" + fileName);
			if (number % 4000 == 0)
				System.out.println("Load " + number + " " + extension
						+ "-files: " + srcDirPath);
		}

		List<String> dirList = PVFile.getDir(srcDirPath);
		Iterator<String> iter2 = dirList.iterator();
		while (iter2.hasNext()) {
			String dir = iter2.next();
			fileList.addAll(getRecursiveFiles(srcDirPath + "/" + dir, extension));
		}
		return fileList;
	}

	/**
	 * 
	 * @param readResult
	 * @param fileName
	 */
	public static List<String> readFile(String fileName) {
		if (!PVFile.isExist(fileName))
			return null;
		List<String> readResult = new ArrayList<String>();
		// try-catch block is used to catch any possible exception when
		// executing this program,
		// such as maybe the file named fileName doesn't exist at all.
		try {
			// FileReader is a class extendting InputStreamReader, and
			// InputStreamReader extends Reader.
			// Reader is the argument of RufferedReader(Reader in).
			// # FileReader() is a connection stream for characters, that
			// connects to a text file.
			// # BufferedReader can be viewed as a buffer used for higher
			// efficiency.
			FileReader fr = new FileReader(fileName);
			BufferedReader in = new BufferedReader(fr);
			String line;
			// read the text one line after another line, until no more line to
			// be read.
			while ((line = in.readLine()) != null) {
				// concatenate lines
				readResult.add(line);
			}
			in.close();
			fr.close();
			return readResult;
		} catch (Exception e) {
			// as long as JVM encounters an exception when executing the
			// program,
			// this catch(){} will catch it and do something.
			System.err.print(e);
		}
		return null;
	}

	public static void print2File(byte[] bytes, String filePath)
	{
		List<String> list = new ArrayList<String>();
		for(int i = 0;i<bytes.length;i++)
			list.add(String.valueOf(bytes[i]));
		print2File(list, filePath);
	}
	
	public static File print2File(List list, String filePath) {
		File file = PVFile.checkCreateFile(filePath);
		try {
			FileWriter fw = new FileWriter(filePath);
			BufferedWriter bw = new BufferedWriter(fw);
			Iterator it = list.iterator();
			while (it.hasNext()) {
				String xyz = it.next().toString();
				bw.write(xyz);
				bw.newLine();
			}
			bw.close();
			fw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return file;
	}

	public static void showProgress(double initLogTime, int i, int size,
			String fileName) {
		String currentTime = getDateTime("HH:mm", new Date());
		long currentTimeValue = System.currentTimeMillis() / 1000;
		System.out.println(currentTime + " : already "
				+ (currentTimeValue - initLogTime) + " sec passed, (" + i + "/"
				+ size + "): " + fileName);
	}

	public static final String getDateTime(String aMask, Date aDate) {
		SimpleDateFormat df = null;
		String returnValue = "";
		df = new SimpleDateFormat(aMask);
		returnValue = df.format(aDate);
		return (returnValue);
	}

	/**
	 * 
	 * @param partitions
	 *            the number of lines of each partitioned file
	 * @param times
	 *            the first times"th" files will be generated
	 * @param filePath
	 *            the path of file to be read
	 * @param outputDir
	 *            the partioned files will be put in here
	 */
	public static void splitFile(int partitions, int times, String filePath,
			String outputDir) {
		ArrayList<String> readResult = new ArrayList<String>();
		// try-catch block is used to catch any possible exception when
		// executing this program,
		// such as maybe the file named fileName doesn't exist at all.
		try {
			// FileReader is a class extendting InputStreamReader, and
			// InputStreamReader extends Reader.
			// Reader is the argument of RufferedReader(Reader in).
			// # FileReader() is a connection stream for characters, that
			// connects to a text file.
			// # BufferedReader can be viewed as a buffer used for higher
			// efficiency.
			BufferedReader in = new BufferedReader(new FileReader(filePath));
			String line;
			int j = 1;
			int k = 0;
			// read the text one line after another line, until no more line to
			// be read.
			for (int i = 0; (line = in.readLine()) != null; i++) {
				// concatenate lines
				readResult.add(line);
				if (i % partitions == 0 && i > 0) {
					k++;
					print2File(readResult, outputDir + "/" + j + ".data");
					System.out.println("outputFile:" + outputDir + "/" + j
							+ ".data");
					j++;
					readResult.clear();
					if (k >= times) {
						System.out.println("done.");
						System.exit(0);
					}
				}
			}
			print2File(readResult, outputDir + "/" + j + ".data");
			j++;
			System.out.println("outputFile:" + outputDir + "/" + j + ".data");
		} catch (Exception e) {
			// as long as JVM encounters an exception when executing the
			// program,
			// this catch(){} will catch it and do something.
			System.err.print(e);
		}
	}

	
	public static Object readZipFile2Object(String zipFilePath) {
		Object obj;
		try {
			double start = System.nanoTime();
			byte[] bytes = (byte[]) PVFile.readBinaryFile(zipFilePath);
			double end = System.nanoTime();
			readTime += (end - start);
			
			double before = System.nanoTime();
			ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
			GZIPInputStream gzipIn = new GZIPInputStream(bais);
			ObjectInputStream objectIn = new ObjectInputStream(gzipIn);
			obj = objectIn.readObject();
			objectIn.close();
			double after = System.nanoTime();
			unzipTime += after - before;
			return obj;
		} catch (Exception e) {
			//e.printStackTrace();
			System.out.println("Error: The input file or data stream is not in SZ format!");
			System.out.println("Please make sure the input data is the byte stream compressed by SZ.");
			System.exit(0);
		}
		return null;
	}
	
	public static Object unGzip(byte[] bytes)
	{
		Object obj = null;
		ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
		try {
			GZIPInputStream gzipIn = new GZIPInputStream(bais);
			ObjectInputStream objectIn = new ObjectInputStream(gzipIn);
			obj = objectIn.readObject();
			objectIn.close();
		} catch (Exception e) {
			System.out.println("Error: The input file or data stream is not in Gzip format!");
			System.out.println("Please check if the input file is correct.");
			System.exit(0);
			//e.printStackTrace();
		}
		return obj;
	}

	/**
	 * Uncompress the incoming file.
	 * 
	 * @param inFileName
	 *            Name of the file to be uncompressed
	 */
	private static void doUncompressFile(String inFileName) {

		try {

			if (!getExtension(inFileName).equalsIgnoreCase("gz")) {
				System.err.println("File name must have extension of \".gz\"");
				System.exit(1);
			}

			System.out.println("Opening the compressed file.");
			GZIPInputStream in = null;
			try {
				in = new GZIPInputStream(new FileInputStream(inFileName));
			} catch (FileNotFoundException e) {
				System.err.println("File not found. " + inFileName);
				System.exit(1);
			}

			System.out.println("Open the output file.");
			String outFileName = getFileName(inFileName);
			FileOutputStream out = null;
			try {
				out = new FileOutputStream(outFileName);
			} catch (FileNotFoundException e) {
				System.err.println("Could not write to file. " + outFileName);
				System.exit(1);
			}

			System.out
					.println("Transfering bytes from compressed file to the output file.");
			byte[] buf = new byte[1024];
			int len;
			while ((len = in.read(buf)) > 0) {
				out.write(buf, 0, len);
			}

			System.out.println("Closing the file and stream");
			in.close();
			out.close();

		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}

	}

	/**
	 * Used to extract and return the extension of a given file.
	 * 
	 * @param f
	 *            Incoming file to get the extension of
	 * @return <code>String</code> representing the extension of the incoming
	 *         file.
	 */
	public static String getExtension(String f) {
		String ext = "";
		int i = f.lastIndexOf('.');

		if (i > 0 && i < f.length() - 1) {
			ext = f.substring(i + 1);
		}
		return ext;
	}

	/**
	 * Used to extract the filename without its extension.
	 * 
	 * @param f
	 *            Incoming file to get the filename
	 * @return <code>String</code> representing the filename without its
	 *         extension.
	 */
	public static String getFileName(String f) {
		String fname = "";
		int i = f.lastIndexOf('.');

		if (i > 0 && i < f.length() - 1) {
			fname = f.substring(0, i);
		}
		return fname;
	}

	public static void append2File(List<String> list, String filePath) {
		PVFile.checkCreateFile(filePath);
		try {
			FileWriter fw = new FileWriter(filePath, true);
			BufferedWriter bw = new BufferedWriter(fw);
			Iterator<String> it = list.iterator();
			while (it.hasNext()) {
				String xyz = it.next();
				bw.write(xyz);
				bw.newLine();
			}
			bw.close();
			fw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Note that the FileWriter must be instantiated with "true" argument.
	 * Example: FileWriter fw = new FileWriter(filePath, true); BufferedWriter
	 * bw = new BufferedWriter(fw); append2File(line,bw); bw.close();
	 * 
	 * @param line
	 * @param fw
	 * @throws IOException
	 */
	public static void append2File(String line, BufferedWriter bw)
			throws IOException {
		bw.append(line);
	}

	public static void writeObject(Object object, String objectFilePath) {
		File file = PVFile.checkCreateFile(objectFilePath);

		try {
			ObjectOutputStream oout = new ObjectOutputStream(
					new FileOutputStream(file));
			oout.writeObject(object);
			oout.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static Object readObject(String objectFilePath) {
		File file = new File(objectFilePath);
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(
					file));
			Object o = in.readObject();
			in.close();
			return o;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		return null;
	}

	public static String checkCompressionRatio(String rootSnapshotDir,
			String outputDir) {
		String aggOriFilePath = rootSnapshotDir + "/aggregateSnapshot.obj";

		File f0 = new File(aggOriFilePath);
		float totalOriSize = f0.length();

		float totalObjSize = 0;
		float totalZipSize = 0;
		List<String> cmpressFileList = PVFile.getFiles(outputDir+"/test");
		Iterator<String> iter = cmpressFileList.iterator();
		while (iter.hasNext()) {
			String fileName = iter.next();
			if (fileName.endsWith("obj")) {
				String filePath = outputDir + "/test/" + fileName;
				File f = new File(filePath);
				totalObjSize += f.length();
			}
		}
		List<String> cmpressFileList2 = PVFile.getFiles(outputDir+"/test2");
		Iterator<String> iter2 = cmpressFileList2.iterator();
		while(iter2.hasNext())
		{
			String fileName = iter2.next();
			if (fileName.endsWith("sz")) {
				String filePath = outputDir + "/test2/" + fileName;
				File f = new File(filePath);	
				totalZipSize += f.length();
			}
		}

		String result = "totalOriSize=" + totalOriSize + " totalObjSize="
				+ totalObjSize + " totalZipSize=" + totalZipSize + " ratio1=1:"
				+ (totalOriSize / totalObjSize) + " ratio2=1:"
				+ (totalOriSize / totalZipSize);
		return result;
	}
	
	public static byte[] readBinaryFile(String filePath)
	{
		File file = new File(filePath);
		return readBinaryFile(file);
	}
	
	/**
	 * 从二进制文件读取字节数组
	 * 
	 * @param sourceFile
	 * @return
	 * @throws IOException
	 */
	public static byte[] readBinaryFile(File sourceFile) {

		if (sourceFile.isFile() && sourceFile.exists()) {
			long fileLength = sourceFile.length();
			if (fileLength > 0L) {
				try {
					BufferedInputStream fis = new BufferedInputStream(
							new FileInputStream(sourceFile));
					byte[] b = new byte[(int) fileLength];

					while (fis.read(b) != -1) {
					}

					fis.close();
					fis = null;

					return b;
				} catch (IOException e) {
					e.printStackTrace();
				}

			}
		}
		return null;
	}

	public static void writeBytes2File(byte[] bytes, String tgtFile)
	{
		BufferedOutputStream bos = null;
			try {
			//create an object of FileOutputStream
			FileOutputStream fos = new FileOutputStream(new File(tgtFile));
	
			//create an object of BufferedOutputStream
			bos = new BufferedOutputStream(fos);
			bos.write(bytes);
			bos.close();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
	
	public static void loadConfig(String configFile)
	{
		List<String> lineList = PVFile.readFile(configFile);
		Iterator<String> iter = lineList.iterator();
		while(iter.hasNext())
		{
			String line = iter.next().trim();
			if(!line.startsWith("#")&&line.contains("="))
			{
				String[] s = line.split("=");
				configMap.put(s[0].trim(), s[1].trim());
			}
		}
	}
	
	/**
	 * Check the endian type for the cpu architecture
	 * @return true (little endian), false (big endian)
	 */
	public static boolean checkEndianType()
	{
		if(ByteOrder.nativeOrder() == ByteOrder.BIG_ENDIAN)
			return false;
		else
			return true;
	}
	
	public static boolean checkVersion(byte[] version)
	{
		for(int i = 0;i<version.length;i++)
			if(version[i]!=versionNumber[i])
				return false;
		return true;
	}
	
	public static void main(String[] args)
	{
		//double[] data = readBinaryFile2DoubleArray("test.dat");
		//System.out.println("data.length="+data.length);
		String userhome = System.getProperty("user.home");
		List<String> list = PVFile.readFile(userhome+"/.bashrc");
		//System.out.println();
	}
}