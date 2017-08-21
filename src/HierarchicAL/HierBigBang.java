package HierarchicAL;

import java.util.ArrayList;

import meka.classifiers.multitarget.CR;
import meka.classifiers.multitarget.meta.BaggingMT;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class HierBigBang {
	
	private Instances FullDataSetHCC;
	private Instances TrainingSetHCC;
	private Instances LabeledSetHCC;
	private Instances UnlabeledSetHCC;
	private Instances TestSetHCC;
	private int BudgetHCC;
	private int HFMIntervalHCC;
	private int TestSetSizeHCC;
	private double SeedSetHCC;
	private int RandomSeedHCC;
	private String SamplingStrategy;
	private Processing processor;
	
	public HierBigBang(Instances FullDataSet,int TestSetSize,double SeedSet,int RandomSeed,int Budget,int HFMInterval,String SamplingStrat){
		
		FullDataSetHCC=FullDataSet;
		TestSetSizeHCC=TestSetSize;
		SeedSetHCC=SeedSet;
		RandomSeedHCC=RandomSeed;
		BudgetHCC=Budget;
		HFMIntervalHCC=HFMInterval;	
		SamplingStrategy=SamplingStrat;		
	}
	
	public void HBBStart() throws Exception{
		
		int BudgetPerLev=BudgetHCC/3;
		
		processor=new Processing();
		Instances[] datasetlist=SetPools(FullDataSetHCC, TestSetSizeHCC, SeedSetHCC, RandomSeedHCC);
		TrainingSetHCC=new Instances(datasetlist[0]);
		TrainingSetHCC.setClassIndex(3);
		LabeledSetHCC=new Instances(datasetlist[1]);
		LabeledSetHCC.setClassIndex(3);
		UnlabeledSetHCC=new Instances(datasetlist[2]);
		UnlabeledSetHCC.setClassIndex(3);
		TestSetHCC=new Instances(datasetlist[3]);
		TestSetHCC.setClassIndex(3);

		//Evaluation();
		System.out.println("Starting Active Learning with Sampling Strategy: " + SamplingStrategy);
		for(int i=1;i<=BudgetPerLev;i++){			
			Instance pickedInstance;
			pickedInstance=processor.GetNextInstance(LabeledSetHCC,UnlabeledSetHCC,"1",SamplingStrategy);
			int ind=UnlabeledSetHCC.indexOf(pickedInstance);
			LabeledSetHCC.add(pickedInstance);
			UnlabeledSetHCC.remove(ind);
			if(i%HFMIntervalHCC==0|| i==BudgetPerLev){
				System.out.println("\nWith " + i + " earned labels for each level:");
				Evaluation();
			}
		}		
	}
	
	public void Evaluation() throws Exception{	

		int[][] resultsheet = new int[3][2];
		Classifier classifier=new NaiveBayes();
		CR cr=new CR();	
		BaggingMT BMT=new BaggingMT();
		
		NGramTokenizer tokenizer = new NGramTokenizer(); 
		tokenizer.setNGramMinSize(1); 
		tokenizer.setNGramMaxSize(1); 
		tokenizer.setDelimiters("\\W");
		
		StringToWordVector stwv=new StringToWordVector();
		stwv.setTokenizer(tokenizer); 
		stwv.setWordsToKeep(1000000); 
		stwv.setDoNotOperateOnPerClassBasis(true); 
		stwv.setLowerCaseTokens(true);
		stwv.setInputFormat(FullDataSetHCC);
		
		FilteredClassifier fc=new FilteredClassifier();
		fc.setFilter(stwv);
		fc.setClassifier(classifier);
		
		cr.setClassifier(fc);
		BMT.setClassifier(cr);
		BMT.buildClassifier(LabeledSetHCC);
		
		for(int j=0;j<TestSetHCC.numInstances();j++){
			double[] resultt=BMT.distributionForInstance(TestSetHCC.instance(j));
			
			if(resultt[0]==TestSetHCC.instance(j).value(0)){
				resultsheet[0][1]=resultsheet[0][1]+1;
			}else{
				resultsheet[0][0]=resultsheet[0][0]+1;
			}
			
			if(resultt[1]==TestSetHCC.instance(j).value(1)){
				resultsheet[1][1]=resultsheet[1][1]+1;				
			}else{
				resultsheet[1][0]=resultsheet[1][0]+1;					
			}
			
			if(TestSetHCC.instance(j).value(2)==0.0 || TestSetHCC.instance(j).value(2)==1.0){
				if(resultt[2]==TestSetHCC.instance(j).value(1)){
					resultsheet[2][1]=resultsheet[2][1]+1;
				}else{
					resultsheet[2][0]=resultsheet[2][0]+1;
				}
			}
			
		}
		processor.AnalyzeResults(resultsheet);
		processor.calculate_HFMeasure(resultsheet);
		
	}
	
	public Instances[] SetPools(Instances FDSet,int TSSize,double Seed,int RSeed) throws Exception{
		
		Instances FDSetRandomized=processor.randomize(processor.TrimDataset(FDSet, "1"),RSeed);
		int TrainSize=(int) Math.round(FDSetRandomized.numInstances()*(1-((double)TSSize/100)));
		int TestSize=FDSetRandomized.numInstances()-TrainSize;
		int SeedSize=(int) Math.round(TrainSize*Seed);
		int UnlabSize=TrainSize-SeedSize;
		
		Instances TrainingSet=new Instances(FDSetRandomized,0,TrainSize);       //Params are Training  Set, Starting index, Number of Instances to copy
		Instances LabeledSet=new Instances(TrainingSet,0,SeedSize);
		Instances UnlabeledSet=new Instances(TrainingSet,SeedSize,UnlabSize);
		Instances TestSet=new Instances(FDSetRandomized,TrainSize,TestSize);
		
		Instances[] DataSetsArray=new Instances[4];
		DataSetsArray[0]=TrainingSet;
		DataSetsArray[1]=LabeledSet;
		DataSetsArray[2]=UnlabeledSet;
		DataSetsArray[3]=TestSet;
		
		return DataSetsArray;						
	}
}
