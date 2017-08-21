package HierarchicAL;

import weka.core.Instance;
import weka.core.Instances;

public class HierSingleFlat extends Abba{
	
	private Instances FullDataSetHSF;
	private Instances TrainingSetHSF;
	private Instances LabeledSetHSF;
	private Instances UnlabeledSetHSF;
	private Instances TestSetHSF;
	private int BudgetHSF;
	private int HFMIntervalHSF;
	private int TestSetSizeHSF;
	private double SeedSetHSF;
	private int RandomSeedHSF;
	private String SamplingStrategy;
	
	public HierSingleFlat(Instances FullDataSet,int TestSetSize,double SeedSet,int RandomSeed,int Budget,int HFMInterval,String SamplingStrat){
		
		FullDataSetHSF=FullDataSet;
		TestSetSizeHSF=TestSetSize;
		SeedSetHSF=SeedSet;
		RandomSeedHSF=RandomSeed;
		HFMIntervalHSF=HFMInterval;
		BudgetHSF=Budget;
		HFMIntervalHSF=HFMInterval;
		SamplingStrategy=SamplingStrat;
	}
	
	public void HSFStart() throws Exception{
		
		Processing processor=new Processing();
		FullDataSetHSF= processor.MergeAttribs(FullDataSetHSF);		
		String[] RemoveIndices={"2,3,4","1,4","1","1,2,3,4"}; 		
		Instances[] datasetlist=processor.SetPools(FullDataSetHSF, TestSetSizeHSF, SeedSetHSF, RandomSeedHSF);
		
		TrainingSetHSF=new Instances(datasetlist[0]);
		TrainingSetHSF.setClassIndex(4);	
		LabeledSetHSF=new Instances(datasetlist[1]);
		LabeledSetHSF.setClassIndex(4);
		UnlabeledSetHSF=new Instances(datasetlist[2]);
		UnlabeledSetHSF.setClassIndex(4);
		TestSetHSF=new Instances(datasetlist[3]);
		TestSetHSF.setClassIndex(4);
		
		System.out.println("Starting Active Learning with Sampling Strategy: " + SamplingStrategy);
		for(int i=1;i<=BudgetHSF;i++){
			Instance pickedInstance;
			pickedInstance=processor.GetNextInstance(LabeledSetHSF,UnlabeledSetHSF,RemoveIndices[0],SamplingStrategy);
			int ind=UnlabeledSetHSF.indexOf(pickedInstance);
			LabeledSetHSF.add(pickedInstance);
			UnlabeledSetHSF.remove(ind);
			if(i%HFMIntervalHSF==0 || i==BudgetHSF){
				System.out.println("\nWith " + i + " earned labels:");
				processor.EvaluationHSF(FullDataSetHSF,LabeledSetHSF,TestSetHSF,RemoveIndices);
			}			
		}
	}
}