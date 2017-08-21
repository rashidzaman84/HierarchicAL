package HierarchicAL;

import java.io.*;
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;

public class UserInput {
	
	public Instances User_Input() {
		
		Instances data = null;
			
		try {
			
			BufferedReader reader=new BufferedReader(new FileReader("datasets/dataset_algo1_ID.arff"));
			data=new Instances(reader);
			reader.close();
			} catch (IOException e) {
				System.out.println("IOException: " + e);
			}
		
		return data;
		}
			
	public String User_Input(String message) {
			
			String input = null;
			System.out.print(message + " ");
		
			try {
			BufferedReader readerr = new BufferedReader(
			new InputStreamReader(System.in));
			input = readerr.readLine();
			readerr.close();
			if (input.length() == 0 ) return null;
			} catch (IOException e) {
			System.out.println("IOException: " + e);
			}
			return input;
			}

}

