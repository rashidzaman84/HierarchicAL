package HierarchicAL;

import java.io.IOException;
import java.util.Arrays;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class HierClassChain extends Abba {
	
	private Instances FullDataSetHCC;
	private Instances TrainingSetHCC;
	private ArrayList<Instances> LabeledSetHCC=new ArrayList<Instances>(3);
	private ArrayList<Instances> UnlabeledSetHCC=new ArrayList<Instances>(3);
	private ArrayList<Instances> TestSetHCC=new ArrayList<Instances>(3);
	private int BudgetHCC;
	private int HFMIntervalHCC;
	private int TestSetSizeHCC;
	private double SeedSetHCC;
	private int RandomSeedHCC;
	private String SamplingStrategy;
		
	public HierClassChain(Instances FullDataSet,int TestSetSize,double SeedSet,int RandomSeed,int Budget,int HFMInterval,String SamplingStrat){
		
		FullDataSetHCC=FullDataSet;
		TestSetSizeHCC=TestSetSize;
		SeedSetHCC=SeedSet;
		RandomSeedHCC=RandomSeed;
		BudgetHCC=Budget;
		HFMIntervalHCC=HFMInterval;	
		SamplingStrategy=SamplingStrat;		
	}
	
	public void HCCStart() throws Exception{
		
		int BudgetPerLev=BudgetHCC/3;
				
		String[] RemoveIndices={"1,3,4","1,4","1"};                    //for Weka Remove Filter the indices start from 1
		
		Processing processor=new Processing();
		Instances[] datasetlist=processor.SetPools(FullDataSetHCC, TestSetSizeHCC, SeedSetHCC, RandomSeedHCC);
		TrainingSetHCC=new Instances(datasetlist[0]);		
		SetLevelPools(datasetlist);
		
		System.out.println("Starting Active Learning with Sampling Strategy: " + SamplingStrategy);
		for(int i=1;i<=BudgetPerLev;i++){			
			Instance pickedInstance;
			pickedInstance=processor.GetNextInstance(LabeledSetHCC.get(0),UnlabeledSetHCC.get(0),RemoveIndices[0],SamplingStrategy);
			int ind=UnlabeledSetHCC.get(0).indexOf(pickedInstance);
			LabeledSetHCC.get(0).add(pickedInstance);
			UnlabeledSetHCC.get(0).remove(ind);
					
			if(i%HFMIntervalHCC==0 || i==BudgetPerLev){
				LabeledSetHCC.set(1, new Instances(datasetlist[1]));
				LabeledSetHCC.get(1).setClassIndex(2);
				UnlabeledSetHCC.set(1, new Instances(LabeledSetHCC.get(0)));
				UnlabeledSetHCC.get(1).setClassIndex(2);
				
				PopulateUnlabPool(0,RemoveIndices[0]);
				
				for(int j=0;j<i;j++){
					pickedInstance = null;
					pickedInstance=processor.GetNextInstance(LabeledSetHCC.get(1),UnlabeledSetHCC.get(1),RemoveIndices[1],SamplingStrategy);
					ind=-1;
					ind=UnlabeledSetHCC.get(1).indexOf(pickedInstance);
					LabeledSetHCC.get(1).add(pickedInstance);
					UnlabeledSetHCC.get(1).remove(ind);				
				}				
				LabeledSetHCC.set(2, new Instances(datasetlist[1]));
				LabeledSetHCC.get(2).setClassIndex(3);
				UnlabeledSetHCC.set(2, new Instances(LabeledSetHCC.get(0)));
				UnlabeledSetHCC.get(2).setClassIndex(3);
				
				PopulateUnlabPool(1,RemoveIndices[1]);
				
				for(int k=0;k<i;k++){
					pickedInstance = null;
					pickedInstance=processor.GetNextInstance(LabeledSetHCC.get(2),UnlabeledSetHCC.get(2),RemoveIndices[2],SamplingStrategy);
					ind=-1;
					ind=UnlabeledSetHCC.get(2).indexOf(pickedInstance);
					LabeledSetHCC.get(2).add(pickedInstance);
					UnlabeledSetHCC.get(2).remove(ind);								
				}
				System.out.println("\nWith " + i + " earned labels for each Level:");
				processor.EvaluationHCC(FullDataSetHCC,LabeledSetHCC,TestSetHCC,RemoveIndices);
			}
		}		
	}			
		
	public void SetLevelPools(Instances[] datasetList){
		
		for(int a=0;a<3;a++){			
			if(a==0){
				LabeledSetHCC.add(new Instances(datasetList[1]));
				LabeledSetHCC.get(a).setClassIndex(a+1);
				UnlabeledSetHCC.add(new Instances(datasetList[2]));
				UnlabeledSetHCC.get(a).setClassIndex(a+1);
				TestSetHCC.add(new Instances(datasetList[3]));
				TestSetHCC.get(a).setClassIndex(a+1);
			}else{
				LabeledSetHCC.add(new Instances(datasetList[1],-1));
				LabeledSetHCC.get(a).setClassIndex(a+1);
				UnlabeledSetHCC.add(new Instances(datasetList[2],-1));
				UnlabeledSetHCC.get(a).setClassIndex(a+1);
				TestSetHCC.add(new Instances(datasetList[3]));
				TestSetHCC.get(a).setClassIndex(a+1);							
			}			
		}		
	}
	
	public void PopulateUnlabPool(int level,String rmindices) throws Exception{
		
		//Here we populate the Unlabled Pool for lower levels suffering from Instances-Scarcity
		FilteredClassifier trick=new FilteredClassifier();
		Classifier nb=new NaiveBayes();
		trick.setClassifier(nb);
		Remove remove=new Remove();
		remove.setAttributeIndices(rmindices);
		trick.setFilter(remove);   
		trick.buildClassifier(LabeledSetHCC.get(level));	
		
		for(int i=0;i<UnlabeledSetHCC.get(level).numInstances();i++){          
			double[] result=trick.distributionForInstance(UnlabeledSetHCC.get(level).instance(i));
			
			if(result[0]>0.8){
				Instance inst=UnlabeledSetHCC.get(level).instance(i);
				inst.setValue(level+1, 0.0);
				UnlabeledSetHCC.get(level+1).add(inst);				
			}
			else if(result[1]>0.8){
				Instance inst=UnlabeledSetHCC.get(level).instance(i);
				inst.setValue(level+1, 1.0);
				UnlabeledSetHCC.get(level+1).add(inst);
			}
		}		
	}
}

