package HierarchicAL;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.Randomize;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;

public class Processing {
		
	public Instances[] SetPools(Instances FDSet,int TSSize,double Seed,int RSeed) throws Exception{
		
		Instances FDSetRandomized=randomize(FDSet,RSeed);
		int TrainSize=(int) Math.round(FDSetRandomized.numInstances()*(1-((double)TSSize/100)));
		int TestSize=FDSetRandomized.numInstances()-TrainSize;
		int SeedSize=(int) Math.round(TrainSize*Seed);
		int UnlabSize=TrainSize-SeedSize;
		
		Instances TrainingSet=new Instances(FDSetRandomized,0,TrainSize);       //Params are Training  Set, Starting index, Number of Instances to copy
		Instances TestSet=new Instances(FDSetRandomized,TrainSize,TestSize);
		
		Instances BOW=BagOfWords(FDSetRandomized,TrainingSet);
		Instances LabeledSet=new Instances(BOW,0,SeedSize);
		Instances UnlabeledSet=new Instances(BOW,SeedSize,UnlabSize);	
		
		Instances[] DataSetsArray=new Instances[4];
		DataSetsArray[0]=TrainingSet;
		DataSetsArray[1]=LabeledSet;
		DataSetsArray[2]=UnlabeledSet;
		DataSetsArray[3]=TestSet;
		
		return DataSetsArray;
						
	}
	
	public static Instances randomize(Instances data, int seed) {
		try {
			
			Randomize rand = new Randomize();
			
			rand.setRandomSeed(seed);
			rand.setInputFormat(data);
			Instances newData = Filter.useFilter(data, rand);
			return newData;
		} catch (Exception e) {
			
			e.printStackTrace();
		}
		return null;
	}
	
	public Instances BagOfWords(Instances Dataset,Instances dataset) throws Exception{
		
		NGramTokenizer tokenizer = new NGramTokenizer(); 
		tokenizer.setNGramMinSize(1); 
		tokenizer.setNGramMaxSize(1); 
		tokenizer.setDelimiters("\\W");
		
		StringToWordVector stwv=new StringToWordVector();
		stwv.setTokenizer(tokenizer); 
		stwv.setWordsToKeep(1000000); 
		stwv.setDoNotOperateOnPerClassBasis(true); 
		stwv.setLowerCaseTokens(true);
		stwv.setInputFormat(Dataset);
					
		Instances BOW = Filter.useFilter(dataset,stwv);
		return BOW;
	}
	
	public Instances MergeAttribs(Instances Dataset) throws Exception{
		
		Instances newData = new Instances(Dataset);
		Add filter= new Add();
		filter.setAttributeIndex("5");
		filter.setNominalLabels("L11,L12,L13,L21,L22,L23");   //{(1)S&E:(11)P,(12)N,(13)Neu....(2)Politics:(21)P,(22)N,(23)N
		filter.setAttributeName("Merged");
		filter.setInputFormat(newData);
		newData = Filter.useFilter(newData, filter);
		
		for(int e=0;e<newData.numInstances();e++){
			if(newData.instance(e).value(1)==0.0){
				if(newData.instance(e).value(2)==0.0){
					if(newData.instance(e).value(3)==0.0){
						newData.instance(e).setValue(4, "L11");
					}else if (newData.instance(e).value(3)==1.0){
						newData.instance(e).setValue(4, "L12");
					}
					
				}else if(newData.instance(e).value(2)==1.0){
					newData.instance(e).setValue(4, "L13");
				}
				
			}else if(newData.instance(e).value(1)==1.0){
				if(newData.instance(e).value(2)==0.0){
					if(newData.instance(e).value(3)==0.0){
						newData.instance(e).setValue(4, "L21");
					}else if (newData.instance(e).value(3)==1.0){
						newData.instance(e).setValue(4, "L22");
					}
					
				}else if(newData.instance(e).value(2)==1.0){
					newData.instance(e).setValue(4, "L23");
				}
				
			}
					
		}
		return newData;
	}
	
	public Instance GetNextInstance(Instances Labelled,Instances Unlabelled, String removind,String SampStrg) throws Exception{
		
		int indexofmaxuncertainty=-1;
		Instance selectedinstance=null;
		Classifier samplingclassifier=new NaiveBayes();
		
		try{
			if(SampStrg.equals("Uncertainty")){
				samplingclassifier.buildClassifier(Labelled);
				Remove remove = new Remove();
				remove.setAttributeIndices(removind);
				FilteredClassifier fc=new FilteredClassifier();
				fc.setFilter(remove);
				fc.setClassifier(samplingclassifier);
				fc.buildClassifier(Labelled);
				Evaluation eval=new Evaluation(Labelled);				
				eval.evaluateModel(fc, Unlabelled);
				double maxuncertainty = Double.MAX_VALUE;
				
				for(int j=0;j<Unlabelled.numInstances();j++){						
					double[] result=fc.distributionForInstance(Unlabelled.instance(j));
					Arrays.sort(result);
					double uncertainty=result[result.length-1]-result[result.length-2];
					if (uncertainty<maxuncertainty){
						maxuncertainty=uncertainty;
						indexofmaxuncertainty=j;
					}
				}
			}else if(SampStrg.equals("Random")){				
				indexofmaxuncertainty=(int) (Math.random() * Unlabelled.size());								
			}			
			
		}catch (IOException e) {
			System.out.println("IOException: " + e);
		}		
		
		selectedinstance=Unlabelled.instance(indexofmaxuncertainty);
		return selectedinstance;		
	}
	
	public void EvaluationHCC(Instances FullDataSet,ArrayList<Instances> Labeled,ArrayList<Instances> Testsett,String[] RemoveIndices) throws Exception {
		
		ArrayList<Instances> Tweetset=new ArrayList<Instances>(3);
		Tweetset=ReconstructDatasets(FullDataSet,Labeled);
		ArrayList<Instances> Testset=new ArrayList<Instances>(3);
		ArrayList<FilteredClassifier> FCs=new ArrayList<FilteredClassifier>(3);
		int[][] resultsheet = new int[3][2];		
		
		for(int x=0;x<3;x++){
			FCs.add(SetFC(Tweetset.get(x),RemoveIndices[x]));
			Testset.add(new Instances(TrimDataset(Testsett.get(x),RemoveIndices[x])));	
		}
		
		for(int y=0;y<Testset.get(0).numInstances();y++){
			double PredLabel1=FCs.get(0).classifyInstance(Testset.get(0).instance(y));
			double PredLabel2=FCs.get(1).classifyInstance(Testset.get(1).instance(y));
			double PredLabel3=FCs.get(2).classifyInstance(Testset.get(2).instance(y));

			if(PredLabel1==Testset.get(0).instance(y).classValue()){
				resultsheet[0][1]=resultsheet[0][1]+1;				
			}else{
				resultsheet[0][0]=resultsheet[0][0]+1;
			}
			
			if(PredLabel2==Testset.get(1).instance(y).classValue()){
				resultsheet[1][1]=resultsheet[1][1]+1;				
			}else{
				resultsheet[1][0]=resultsheet[1][0]+1;					
			}
			
			if(Testset.get(2).instance(y).classValue()==0.0 || Testset.get(2).instance(y).classValue()==1.0){
				if(PredLabel3==Testset.get(2).instance(y).classValue()){
					resultsheet[2][1]=resultsheet[2][1]+1;
				}else{
					resultsheet[2][0]=resultsheet[2][0]+1;
				}
			}			
		}
		
		AnalyzeResults(resultsheet);
		calculate_HFMeasure(resultsheet);
	}
	
	public void EvaluationHSF(Instances FullDataSet,Instances Labeled,Instances Testset,String[] RemoveIndices) throws Exception {
		
		int[][] resultsheet = new int[3][2];
		Instances TweetSet=ReconstructDataset(FullDataSet,Labeled);
		Instances Tweets=new Instances(Labeled);
		Instances Test=new Instances(Testset);
		TweetSet=TrimDataset(TweetSet,RemoveIndices[3]);
		Testset=TrimDataset(Test,RemoveIndices[3]);
		
		NGramTokenizer tokenizer = new NGramTokenizer(); 
		tokenizer.setNGramMinSize(1); 
		tokenizer.setNGramMaxSize(1); 
		tokenizer.setDelimiters("\\W");
		StringToWordVector stwv=new StringToWordVector();
		stwv.setTokenizer(tokenizer); 
		stwv.setWordsToKeep(1000000); 
		stwv.setDoNotOperateOnPerClassBasis(true); 
		stwv.setLowerCaseTokens(true);
		
		Classifier classifier=new NaiveBayes();
		FilteredClassifier fc=new FilteredClassifier();
		fc.setFilter(stwv);
		fc.setClassifier(classifier);
		fc.buildClassifier(TweetSet);
		
		for (int i = 0; i < Testset.numInstances(); i++) {
			double PredLabel= fc.classifyInstance(Testset.instance(i));
			double TrueLabel=Testset.instance(i).classValue();
			double[] PredLabelArray=LabToArray(PredLabel);
			double[] TrueLabelArray=LabToArray(TrueLabel);
			
			for(int m=0;m<TrueLabelArray.length;m++){
				if(m<PredLabelArray.length){
					if(TrueLabelArray[m]==PredLabelArray[m]){
						resultsheet[m][1]=resultsheet[m][1]+1;
					}else{
						resultsheet[m][0]=resultsheet[m][0]+1;
					}					
				}else{
					resultsheet[m][0]=resultsheet[m][0]+1;
				}
			}			
		}
		AnalyzeResults(resultsheet);
		calculate_HFMeasure(resultsheet);
	}
	
	public void EvaluationHTD(Instances FullDataSet,ArrayList<Instances> Labeled,ArrayList<Instances> Testsett,String[] RemoveIndices) throws Exception {
		
		int[][] resultsheet = new int[3][2];
		ArrayList<Instances> Tweetset=new ArrayList<Instances>(5);
		Tweetset=ReconstructDataset2(FullDataSet,Labeled);	
		ArrayList<Instances> Testset=new ArrayList<Instances>(5);
		ArrayList<FilteredClassifier> FCs=new ArrayList<FilteredClassifier>(5);
		
		RemoveIndices=new String[]{"1,3,4","1,2,4","1,2,4","1,2,3","1,2,3"};
						
		for(int x=0;x<5;x++){
			FCs.add(SetFC(Tweetset.get(x),RemoveIndices[x]));
			Testset.add(new Instances(TrimDataset(Testsett.get(x),RemoveIndices[x])));
		}
		for(int y=0;y<Testset.get(0).numInstances();y++){

			double PredLabel1=FCs.get(0).classifyInstance(Testset.get(0).instance(y));
			
			if(PredLabel1==Testset.get(0).instance(y).classValue()){
				resultsheet[0][1]=resultsheet[0][1]+1;
				if(PredLabel1==0.0){
					double PredLabel2=FCs.get(1).classifyInstance(Testset.get(1).instance(y));
					if(PredLabel2==Testset.get(1).instance(y).classValue()){
						resultsheet[1][1]=resultsheet[1][1]+1;
						if(PredLabel2==0.0){
							double PredLabel3=FCs.get(3).classifyInstance(Testset.get(3).instance(y));
							if(PredLabel3==Testset.get(3).instance(y).classValue()){
								resultsheet[2][1]=resultsheet[2][1]+1;
							}else{
								resultsheet[2][0]=resultsheet[2][0]+1;
							}
						}
					}else{
						resultsheet[1][0]=resultsheet[1][0]+1;
						if(PredLabel2==0.0){
							double PredLabel3=FCs.get(3).classifyInstance(Testset.get(3).instance(y));
							if(PredLabel3==Testset.get(3).instance(y).classValue()){
								resultsheet[2][1]=resultsheet[2][1]+1;
							}else{
								resultsheet[2][0]=resultsheet[2][0]+1;
							}
						}
					}
				}else{
					double PredLabel2=FCs.get(2).classifyInstance(Testset.get(2).instance(y));
					if(PredLabel2==Testset.get(1).instance(y).classValue()){
						resultsheet[1][1]=resultsheet[1][1]+1;
						if(PredLabel2==0.0){
							double PredLabel3=FCs.get(3).classifyInstance(Testset.get(3).instance(y));
							if(PredLabel3==Testset.get(3).instance(y).classValue()){
								resultsheet[2][1]=resultsheet[2][1]+1;
							}else{
								resultsheet[2][0]=resultsheet[2][0]+1;
							}
						}
					}else{
						resultsheet[1][0]=resultsheet[1][0]+1;
						if(PredLabel2==0.0){
							double PredLabel3=FCs.get(3).classifyInstance(Testset.get(3).instance(y));
							if(PredLabel3==Testset.get(3).instance(y).classValue()){
								resultsheet[2][1]=resultsheet[2][1]+1;
							}else{
								resultsheet[2][0]=resultsheet[2][0]+1;
							}
						}
					}
				}			
						
			}else{
				resultsheet[0][0]=resultsheet[0][0]+1;
				if(PredLabel1==0.0){
					double PredLabel2=FCs.get(1).classifyInstance(Testset.get(1).instance(y));
					if(PredLabel2==Testset.get(1).instance(y).classValue()){
						resultsheet[1][1]=resultsheet[1][1]+1;
						if(PredLabel2==0.0){
							double PredLabel3=FCs.get(3).classifyInstance(Testset.get(3).instance(y));
							if(PredLabel3==Testset.get(3).instance(y).classValue()){
								resultsheet[2][1]=resultsheet[2][1]+1;
							}else{
								resultsheet[2][0]=resultsheet[2][0]+1;
							}
						}
					}else{
						resultsheet[1][0]=resultsheet[1][0]+1;
						if(PredLabel2==0.0){
							double PredLabel3=FCs.get(3).classifyInstance(Testset.get(3).instance(y));
							if(PredLabel3==Testset.get(3).instance(y).classValue()){
								resultsheet[2][1]=resultsheet[2][1]+1;
							}else{
								resultsheet[2][0]=resultsheet[2][0]+1;
							}
						}
					}
				}else{
					double PredLabel2=FCs.get(2).classifyInstance(Testset.get(2).instance(y));
					if(PredLabel2==Testset.get(1).instance(y).classValue()){
						resultsheet[1][1]=resultsheet[1][1]+1;
						if(PredLabel2==0.0){
							double PredLabel3=FCs.get(3).classifyInstance(Testset.get(3).instance(y));
							if(PredLabel3==Testset.get(3).instance(y).classValue()){
								resultsheet[2][1]=resultsheet[2][1]+1;
							}else{
								resultsheet[2][0]=resultsheet[2][0]+1;
							}
						}
					}else{
						resultsheet[1][0]=resultsheet[1][0]+1;
						if(PredLabel2==0.0){
							double PredLabel3=FCs.get(3).classifyInstance(Testset.get(3).instance(y));
							if(PredLabel3==Testset.get(3).instance(y).classValue()){
								resultsheet[2][1]=resultsheet[2][1]+1;
							}else{
								resultsheet[2][0]=resultsheet[2][0]+1;
							}
						}
					}
				}							
			}
		}		
		AnalyzeResults(resultsheet);
		calculate_HFMeasure(resultsheet);		
	}		
	
	public ArrayList ReconstructDatasets(Instances FullDataSet,ArrayList<Instances>Labeled){
		
		ArrayList<Instances> LabeledSet_Tweets=new ArrayList<Instances>(3);
		
		for(int l=0;l<3;l++){
			double indx=0;
			Instance inst=null;
			LabeledSet_Tweets.add(new Instances(FullDataSet,-1));
			LabeledSet_Tweets.get(l).setClassIndex(l+1);
			for(int m=0;m<Labeled.get(l).numInstances();m++){
				
				indx=Labeled.get(l).instance(m).value(0);
				LabeledSet_Tweets.get(l).add(FullDataSet.instance((int)indx-1));							
			}
		}
		return LabeledSet_Tweets;
	}
	
	public Instances ReconstructDataset(Instances FullDataSet,Instances Labeled){
		
		double indx=0;
		Instance inst=null;
		Instances LabeledSet_Tweets=new Instances(FullDataSet,-1);
		LabeledSet_Tweets.setClassIndex(4);
		for(int m=0;m<Labeled.numInstances();m++){
			indx=Labeled.instance(m).value(0);
			LabeledSet_Tweets.add(FullDataSet.instance((int)indx-1));
		}
		return LabeledSet_Tweets;
	}
	
	public ArrayList ReconstructDataset2(Instances FullDataSet,ArrayList<Instances>Labeled){
				
		ArrayList<Instances> LabeledSet_Tweets=new ArrayList<Instances>(5);
				
		for(int l=0;l<5;l++){
			double indx=0;
			Instance inst=null;
			LabeledSet_Tweets.add(new Instances(FullDataSet,-1));
			LabeledSet_Tweets.get(l).setClassIndex(Labeled.get(l).classIndex());
			
			for(int m=0;m<Labeled.get(l).numInstances();m++){				
				indx=Labeled.get(l).instance(m).value(0);
				LabeledSet_Tweets.get(l).add(FullDataSet.instance((int)indx-1));							
			}
			LabeledSet_Tweets.get(l).setClassIndex(Labeled.get(l).classIndex());
		}
		return LabeledSet_Tweets;
	}
	
	public FilteredClassifier SetFC(Instances Tweets,String rmindices) throws Exception{
		
		Instances Tweets_filtered=TrimDataset(Tweets,rmindices);
		
		NGramTokenizer tokenizer = new NGramTokenizer(); 
		tokenizer.setNGramMinSize(1); 
		tokenizer.setNGramMaxSize(1); 
		tokenizer.setDelimiters("\\W");
				
		StringToWordVector stwv=new StringToWordVector();
		stwv.setTokenizer(tokenizer); 
		stwv.setWordsToKeep(1000000); 
		stwv.setDoNotOperateOnPerClassBasis(true); 
		stwv.setLowerCaseTokens(true);
		stwv.setInputFormat(Tweets_filtered);
				
		FilteredClassifier fc=new FilteredClassifier();
		Classifier classifier=new NaiveBayes();
		fc.setClassifier(classifier);
		fc.setFilter(stwv);
		fc.buildClassifier(Tweets_filtered);
		
		return fc;		
	}
	
	public Instances TrimDataset(Instances Tweets,String rmindices) throws Exception{
		
		Remove rm=new Remove();
		rm.setAttributeIndices(rmindices);
		rm.setInputFormat(Tweets);
		Instances Tweets_filtered=Filter.useFilter(Tweets, rm);
		return Tweets_filtered;
	}
	
	public void AnalyzeResults(int[][] result){
		System.out.println("Level 1 Correctly classified : " + result[0][1] + " & Mis-classified : " + result[0][0]);
		System.out.println("Level 2 Correctly classified : " + result[1][1] + " & Mis-classified : " + result[1][0]);
		System.out.println("Level 3 Correctly classified : " + result[2][1] + " & Mis-classified : " + result[2][0]);
	}
	
	public void calculate_HFMeasure(int[][] result){
		int total=0;
		int correct=0;
		for(int j=0;j<3;j++){
			correct+=result[j][1];
			total+=result[j][0]+result[j][1];
		}
		double hprecision=(double)correct/(double)(total);
		double hrecall=(double)correct/(double)(total);
		double hfmeasure=(2*hprecision*hrecall)/(hprecision+hrecall);
		System.out.print("The Hierarchical F-measure for current settings is : " + hfmeasure + "\n");
		System.out.flush();
	}
	
	public double[] LabToArray(double Label){
		
		double[] arrray=null;
		switch((int)Label){
			
		case 0:
			arrray=new double[] {0.0,0.0,0.0};
			break;
		case 1:
			arrray=new double[] {0.0,0.0,1.0};
			break;
		case 2:
			arrray=new double[] {0.0,1.0};
			break;
		case 3:
			arrray=new double[] {1.0,0.0,0.0};
			break;
		case 4:
			arrray=new double[] {1.0,0.0,1.0};
			break;
		case 5:
			arrray=new double[] {1.0,1.0};
			break;		
		}
		return arrray;		
	}
}
