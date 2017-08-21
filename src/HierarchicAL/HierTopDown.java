package HierarchicAL;

import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

public class HierTopDown extends Abba {
	
	private Instances FullDataSetHTD;
	private Instances TrainingSetHTD;
	private ArrayList<Instances> LabeledSetHTD=new ArrayList<Instances>(5);
	private ArrayList<Instances> UnlabeledSetHTD=new ArrayList<Instances>(5);
	private ArrayList<Instances> TestSetHTD=new ArrayList<Instances>(5);
	private int BudgetHTD;
	private int HFMIntervalHTD;
	private int TestSetSizeHTD;
	private double SeedSetHTD;
	private int RandomSeedHTD;
	private String SamplingStrategy;
	
	public HierTopDown(Instances FullDataSet,int TestSetSize,double SeedSet,int RandomSeed,int Budget,int HFMInterval,String SamplingStrat){
		
		FullDataSetHTD=FullDataSet;
		TestSetSizeHTD=TestSetSize;
		SeedSetHTD=SeedSet;
		RandomSeedHTD=RandomSeed;
		BudgetHTD=Budget;
		HFMIntervalHTD=HFMInterval;	
		SamplingStrategy=SamplingStrat;		
	}
	
	public void HTDStart() throws Exception{
		
		int BudgetPerLev=BudgetHTD/3;
		String[] RemoveIndices={"1,3,4","1,2,4","1,2,3"};
		Processing processor=new Processing();
		Instances[] datasetlist=processor.SetPools(FullDataSetHTD, TestSetSizeHTD, SeedSetHTD, RandomSeedHTD);
		TrainingSetHTD=new Instances(datasetlist[0]);
		SetLevelPools(datasetlist);
		Instance pickedInstance;
		
		System.out.println("Starting Active Learning with Sampling Strategy: " + SamplingStrategy);
		for(int i=1;i<=BudgetPerLev;i++){			
			pickedInstance=processor.GetNextInstance(LabeledSetHTD.get(0),UnlabeledSetHTD.get(0),RemoveIndices[0],SamplingStrategy);
			int ind=UnlabeledSetHTD.get(0).indexOf(pickedInstance);
			LabeledSetHTD.get(0).add(pickedInstance);
			if(pickedInstance.classValue()==0.0){				
				UnlabeledSetHTD.get(1).add(pickedInstance);				
			}
			else{
				UnlabeledSetHTD.get(2).add(pickedInstance);
			}
			UnlabeledSetHTD.get(0).remove(ind);
			
			if(i%HFMIntervalHTD==0 || i==BudgetPerLev){
				PopulateLabeledPools(LabeledSetHTD.get(0),1,2,2);
				PopulateUnlabPools(0,RemoveIndices[0]);				
				for(int j=0;j<i/2;j++){
					pickedInstance = null;
					pickedInstance=processor.GetNextInstance(LabeledSetHTD.get(1),UnlabeledSetHTD.get(1),RemoveIndices[1],SamplingStrategy);
					ind=-1;
					ind=UnlabeledSetHTD.get(1).indexOf(pickedInstance);
					LabeledSetHTD.get(1).add(pickedInstance);
					UnlabeledSetHTD.get(1).remove(ind);	
				}
				
				for(int k=0;k<i/2;k++){
					pickedInstance = null;
					pickedInstance=processor.GetNextInstance(LabeledSetHTD.get(2),UnlabeledSetHTD.get(2),RemoveIndices[1],SamplingStrategy);
					ind=-1;
					ind=UnlabeledSetHTD.get(2).indexOf(pickedInstance);
					LabeledSetHTD.get(2).add(pickedInstance);
					UnlabeledSetHTD.get(2).remove(ind);	
				}
				
				PopulateLabeledPools(LabeledSetHTD.get(1),3,5,3);
				PopulateLabeledPools(LabeledSetHTD.get(2),4,6,3);
				PopulateUnlabPools(1,RemoveIndices[1]);
				PopulateUnlabPools(2,RemoveIndices[1]);
				
				for(int l=0;l<i/2;l++){
					pickedInstance = null;
					pickedInstance=processor.GetNextInstance(LabeledSetHTD.get(3),UnlabeledSetHTD.get(3),RemoveIndices[2],SamplingStrategy);
					ind=-1;
					ind=UnlabeledSetHTD.get(3).indexOf(pickedInstance);
					LabeledSetHTD.get(3).add(pickedInstance);
					UnlabeledSetHTD.get(3).remove(ind);								
				}
				
				for(int m=0;m<i/2;m++){
					pickedInstance = null;
					pickedInstance=processor.GetNextInstance(LabeledSetHTD.get(4),UnlabeledSetHTD.get(4),RemoveIndices[2],SamplingStrategy);
					ind=-1;
					ind=UnlabeledSetHTD.get(4).indexOf(pickedInstance);
					LabeledSetHTD.get(4).add(pickedInstance);
					UnlabeledSetHTD.get(4).remove(ind);	
				}	
				System.out.println("\nWith " + i + " earned labels per Level:");
				processor.EvaluationHTD(FullDataSetHTD,LabeledSetHTD,TestSetHTD,RemoveIndices);				
			}			
		}
	}
		
	public void SetLevelPools(Instances[] datasetList){
		
		int[] indexarray={1,2,2,3,3};	
		for(int l=0;l<5;l++){
			if(l==0){
				LabeledSetHTD.add(new Instances(datasetList[indexarray[0]]));
				LabeledSetHTD.get(l).setClassIndex(indexarray[0]);
				UnlabeledSetHTD.add(new Instances(datasetList[indexarray[2]]));
				UnlabeledSetHTD.get(l).setClassIndex(indexarray[0]);
				TestSetHTD.add(new Instances(datasetList[indexarray[3]]));
				TestSetHTD.get(l).setClassIndex(indexarray[0]);				
			}else{
				LabeledSetHTD.add(new Instances(datasetList[indexarray[0]],-1));
				LabeledSetHTD.get(l).setClassIndex(indexarray[l]);
				UnlabeledSetHTD.add(new Instances(datasetList[indexarray[2]],-1));
				UnlabeledSetHTD.get(l).setClassIndex(indexarray[l]);	
				TestSetHTD.add(new Instances(datasetList[indexarray[3]]));
				TestSetHTD.get(l).setClassIndex(indexarray[l]);
				
			}
		}
					
	}
	
	private void PopulateLabeledPools(Instances parent, int i, int j, int k) {
		
		for(int u=0;u<parent.numInstances();u++){
			if(parent.instance(u).classValue()==0.0){ 
				LabeledSetHTD.get(i).add(parent.instance(u));
				LabeledSetHTD.get(i).setClassIndex(k);
			}else if(j<5){
				LabeledSetHTD.get(j).add(parent.instance(u));
				LabeledSetHTD.get(j).setClassIndex(k);
			}
		}		
	}
	
	public void PopulateUnlabPools(int level,String rmindices) throws Exception{
		
		//Here we populate the Unlabeled Pool(s) for lower levels suffering from Instances-Scarcity
		FilteredClassifier trick=new FilteredClassifier();
		Classifier nb=new NaiveBayes();
		trick.setClassifier(nb);
		Remove remove=new Remove();
		remove.setAttributeIndices(rmindices);
		trick.setFilter(remove);  
		trick.buildClassifier(LabeledSetHTD.get(level));	
		
		for(int i=0;i<UnlabeledSetHTD.get(level).numInstances();i++){          
			double[] result=trick.distributionForInstance(UnlabeledSetHTD.get(level).instance(i));
			
			if(result[0]>0.8){
				if(level==0){
					Instance inst=UnlabeledSetHTD.get(level).instance(i);
					inst.setValue(level+1, 0.0);
					UnlabeledSetHTD.get(level+1).add(inst);
				}else if(!(level==0)){
					Instance inst=UnlabeledSetHTD.get(level).instance(i);
					inst.setValue(level+1, 0.0);
					UnlabeledSetHTD.get(level+2).add(inst);						
				}								
			}
			else if(result[1]>0.8 && level==0){
				Instance inst=UnlabeledSetHTD.get(level).instance(i);
				inst.setValue(level+1, 1.0);
				UnlabeledSetHTD.get(level+2).add(inst);
			}
		}		
	}	
}