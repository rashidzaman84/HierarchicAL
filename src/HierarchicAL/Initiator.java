package HierarchicAL;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

public class Initiator {
	
	private Instances FullDataset;
	private double SeedSet;          
	private int[] RandomSeed;
	private int Budget;
	private int HFMInterval;
	private int TestSetSize;
	private String SamplingStrategy;
	
	public static void main(String[] args) throws Exception {
			
			Initiator self=new Initiator(); 
			self.Initialize();	
			self.CallHierarMech(1);		//HBB=1, HSF=2, HCC=3, HTD=4	
	}
		
	public void Initialize(){
		
		UserInput UI=new UserInput();
		FullDataset=UI.User_Input();		
		TestSetSize=10;                 //10% of the dataset will be used for Evaluation purposes.
		SeedSet=0.01;                  //Three Seed values used i.e., {0.001,0.005,0.01} of the training set    
		RandomSeed=new int[]{5,89,50,21,42,63,34,14,77,9};
		Budget=100;
		HFMInterval=5;                 //Hierarchical F-meaasure to be calculated after buying this number of instances
		SamplingStrategy="Uncertainty";             //Other option is "Random"		
	}
	
	public void CallHierarMech(int approach) throws Exception{
		
		switch(approach){
		
		case 1:
			HierBigBang hbb=new HierBigBang(FullDataset,TestSetSize,SeedSet,RandomSeed[0],Budget,HFMInterval,"Random");
			hbb.HBBStart();
			break;
		case 2:
			HierSingleFlat hsf=new HierSingleFlat(FullDataset,TestSetSize,SeedSet,RandomSeed[0],Budget,HFMInterval,SamplingStrategy);
			hsf.HSFStart();
			break;
		case 3:			
			HierClassChain hcc=new HierClassChain(FullDataset,TestSetSize,SeedSet,RandomSeed[0],Budget,HFMInterval,SamplingStrategy);
			hcc.HCCStart();
			break;
		case 4:
			HierTopDown htd=new HierTopDown(FullDataset,TestSetSize,SeedSet,RandomSeed[0],Budget,HFMInterval,SamplingStrategy);
			htd.HTDStart();
			break;
		default:
			System.out.println("Wrong choice of Hierarchical Mechanism");					
		}				
	}
}