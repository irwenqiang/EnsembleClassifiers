package cn.edu.xmu.dm.d3c.utils;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class InstanceUtil {
	//
	public String jarName;
	//
	public String jarPath;
	//
	/*
	 * 通过文件名的字符串获得实例
	 */
	public static Instances getInstances(String filename) throws Exception {
		File file = new File(filename);
		return getInstances(file);
	}

	/*
	 * 通过File类获得实例
	 */
	public static Instances getInstances(File file) throws Exception {
		Instances inst = null;
		try {
			ArffLoader loader = new ArffLoader();
			loader.setFile(file);
			inst = loader.getDataSet();
		} catch (Exception e) {
			throw new Exception(e.getMessage());
		}
		return inst;
	}

	/*
	 * 打印出得到的实例
	 */
	public static void printInstances(Instances ins) {
		for (int i = 0; i < ins.numInstances(); i++) {
			System.out.println(ins.instance(i));
		}
	}
	
	/*
	 * 获得jar包路径
	 */
	public void getJarPath(Class clazz)throws Exception{
		//
		String path=clazz.getProtectionDomain().getCodeSource().getLocation().getFile();
		//
		path=java.net.URLDecoder.decode(path, "UTF-8");
		//
		java.io.File jarFile=new java.io.File(path);
		//
		this.jarName=jarFile.getName();
		//
		java.io.File parent=jarFile.getParentFile();
		//
		if(parent!=null){
			//
			this.jarPath=parent.getAbsolutePath();
		}
	}
	
	/*
	 * 
	 */
	public void createClassifyResultFile(int num,List<List<Integer>> classifyRightOrWrong){
		int i,j;
		try {
			//
			//URL url=this.getClass().getResource("/");
			//String path=url.getFile();
			//
			//System.out.println(path);
			//
			getJarPath(InstanceUtil.class);
			
			String  pathPrefix = "D:/tmp/";
			
			File dir = new File(pathPrefix);
			if (!dir.isDirectory()) {
				   dir.mkdir();
			}
			
			String tempPath=pathPrefix+"ClassifyResult.arff";
			//
			System.out.println(tempPath);
			//
			BufferedWriter writer = new BufferedWriter(new FileWriter(tempPath));
			//BufferedWriter writer = new BufferedWriter(new FileWriter("D:/qc/javawork/LibD3C/lib/ClassifyResult.arff"));
			//
			String content=new String();
			content="@relation ClassifyResult";
			writer.write(content);
			writer.newLine();
			for(i=1;i<=num;i++){
				content=new String();
				content="@attribute	"+"A"+i+"	"+"{0,1}";
				writer.write(content);
				writer.newLine();
			}
			content=new String();
			content="@data";
			writer.write(content);
			//
			for(i=0;i<classifyRightOrWrong.size();i++){
				writer.newLine();
				content=new String();
				for(j=0;j<classifyRightOrWrong.get(i).size();j++){
					//
					if(j==0){
						content=classifyRightOrWrong.get(i).get(j).toString();
					}
					else{
						content=content+","+classifyRightOrWrong.get(i).get(j);
					}
				}
				//自己设置一个类别，不过这个类别对结果没什么影响
				writer.write(content);
			}
			writer.flush();
			writer.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	//获取当前时间
	public static String getCurrentTime(){
		Date currentTime=new Date();
		SimpleDateFormat formatter=new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		String dateString=formatter.format(currentTime);
		return dateString;
	}
	//获得时间间隔
	public static long timeCompare(String t1,String t2){
		//
		SimpleDateFormat formatter=new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		//Calendar c1=Calendar.getInstance();
		//Calendar c2=Calendar.getInstance();
		
			//c1.setTime(formatter.parse(t1));
			//c2.setTime(formatter.parse(t2));
		Date d1=new Date();
		Date d2=new Date();
		
		try{
			 d1=formatter.parse(t1);
			 d2=formatter.parse(t2);
		}catch(Exception e){
			e.printStackTrace();
		}
		long result=(d2.getTime()-d1.getTime())/1000;
		
		return result;
		
	}
	//
	public double findTheMaxNo(double[] distributeForInstance){
		//
		int i;
		//
		double No;
		double max;
		//
		List<Double> temp=new ArrayList<Double>();
	
		for(i=0;i<distributeForInstance.length;i++){
			//
			temp.add(distributeForInstance[i]);
		}
		//
		max=Collections.max(temp);
		//
		No=temp.indexOf(max);
		//
		return No;
	}
	
}
