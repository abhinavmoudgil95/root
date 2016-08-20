#include "TMath.h"
#include "TVectorD.h"
#include "TFile.h"
#include "TTree.h"
#include "TMatrix.h"

#include "TMVA/VarTransformHandler.h"
#include "TMVA/DataSet.h"
#include "TMVA/DataSetInfo.h"
#include "TMVA/Event.h"
#include "TMVA/Tools.h"
#include "TMVA/Types.h"
#include "TMVA/MethodBase.h"
#include "TMVA/MethodDNN.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/DataLoader.h"
#include "TMVA/ClassifierFactory.h"
#include "TMVA/DataInputHandler.h"

#include <vector>
#include <iomanip>

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::VarTransformHandler::VarTransformHandler( DataSetInfo& dsi, DataLoader* dl ) 
   : fDataSetInfo(dsi),
     fDataLoader (dl),
     fLogger     ( new MsgLogger(TString("VarTransformHandler").Data(), kINFO) )
{
   // produce one entry for each class and one entry for all classes. If there is only one class, 
   // produce only one entry
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::VarTransformHandler::~VarTransformHandler() 
{
	// do something
   delete fLogger;	
}

////////////////////////////////////////////////////////////////////////////////
/// Autoencoder Transform

TMVA::DataLoader* TMVA::VarTransformHandler::AutoencoderTransform(TString dnnOptions, TString preTrngValue, Int_t indexLayer)
{
   // prepare new loader for DNN training
   Log() << kINFO << "Preparing DataLoader for Autoencoder Transform DNN Training" << Endl;
   TMVA::DataLoader *tempLoader = new TMVA::DataLoader("ae_transform_dataset");
   std::vector<VariableInfo>& vars = fDataSetInfo.GetVariableInfos();
   const UInt_t nvars = fDataSetInfo.GetNVariables();
   Log() << kINFO << nvars << Endl;
   for (UInt_t ivar=0; ivar<nvars; ivar++) {
      tempLoader->AddVariable(vars[ivar].GetExpression(), vars[ivar].GetVarType());
      tempLoader->AddTarget(vars[ivar].GetExpression());
   }
   TTree *data = MakeDataSetTree();
   Log() << kINFO << "Adding regression tree" << Endl;
   tempLoader->AddRegressionTree(data, 1.0);
   tempLoader->PrepareTrainingAndTestTree("","SplitMode=random:!V");

   // extract names contained in "trOptions"
   Types::EMVA theMethod = TMVA::Types::kDNN;
   TString theMethodName = Types::Instance().GetMethodName( theMethod );
   TString JobName = "TMVARegression";

   // book DNN Method
   Event::SetIsTraining(kTRUE);
   gSystem->MakeDirectory(tempLoader->GetName());
   Types::EAnalysisType fAnalysisType = Types::kRegression;
   TString methodTitle = "DNN";
   IMethod* im;
   Log() << kINFO << "****1****" << Endl;   
   im = ClassifierFactory::Instance().Create( std::string(theMethodName),
                                         JobName,
                                         methodTitle,
                                         tempLoader->GetDataSetInfo(),
                                         dnnOptions );
   MethodDNN *method = dynamic_cast<MethodDNN*>(im);
   if (method==0)
   {
      Log() << kINFO << "------------------------method = 0----------------------------" << Endl;
      return fDataLoader;
   }
   Log() << kINFO << "****2****" << Endl;      
   method->SetAnalysisType( fAnalysisType );
      Log() << kINFO << "****3****" << Endl;   
   method->SetupMethod();
      Log() << kINFO << "****4****" << Endl;   
   method->ParseOptions();
      Log() << kINFO << "****5****" << Endl;   
   method->ProcessSetup();
      Log() << kINFO << "****6****" << Endl;   
   method->CheckSetup();

   Log() << kINFO << "****7****" << Endl;   
   // train DNN Method
   method->TrainMethod();
   Log() << kINFO << "Training finished" << Endl;

   // TransformAEEvents
   const std::vector<Event*>& events = fDataSetInfo.GetDataSet()->GetEventCollection();
   // get number of variables for new DataLoader
   TMVA::DataLoader *transformedLoader = new TMVA::DataLoader(fDataSetInfo.GetName());
   const Event* ev = events[0];
   std::vector<Float_t>& tranfValues = method->GetLayerActivationValues(ev, indexLayer);
   UInt_t numOfTranfVariables = tranfValues.size();

   // create a new dataset file
   TString newDataSetName = fDataSetInfo.GetName();
   newDataSetName += "_ae_transformed.root";
   TFile *f = new TFile(newDataSetName,"RECREATE");

   // get number of classes
   const UInt_t numOfClasses = fDataSetInfo.GetNClasses();
   const UInt_t numOfTargets = fDataSetInfo.GetNTargets();
   const UInt_t numOfVariables = fDataSetInfo.GetNVariables();
   const UInt_t nevts = events.size();

   Log() << kINFO << "[AE Transform] Total number of events: " << nevts << Endl;

   TString varName, tarName, varType, tarType;
   // it's a regression problem
   // TODO: Add regression trees with weights
   if (numOfTargets != 0)
   {
      Log() << kINFO << "[AE Transform] Number of targets: " << numOfTargets << Endl;
      // create a new tree with transformed variables and original targets
      TTree *R = new TTree("ae_transformed_regtree","AE Transformed Regression Tree");
      for (UInt_t i = 0; i < numOfTranfVariables; i++) {
         varName = "ae_transformed_var";
         varName += i;
         varType = varName;
         varType += "/F";

         Log() << kINFO << "[AE Transform] Adding transformed variable " << varName << " to new DataLoader" << Endl;
         R->Branch(varName, &tranfValues[i], varType);
         transformedLoader->AddVariable(varName, 'F');
      }
      std::vector<VariableInfo>& tars = fDataSetInfo.GetTargetInfos();
      std::vector<Float_t> targets(numOfTargets);
      for (UInt_t i = 0; i < numOfTargets; i++) {
         tarType = tars[i].GetExpression();
         tarType += "/F";

         Log() << kINFO << "[AE Transform] Adding target variable " << tars[i].GetExpression() << " to new DataLoader" << Endl;
         R->Branch(tars[i].GetExpression(), &targets[i], tarType);
         transformedLoader->AddTarget(tars[i].GetExpression());
      }

      // loop over all events, tranform and add to tree
      UInt_t itgt;
      for (UInt_t ievt = 0; ievt < nevts; ievt++) {
         ev = events[ievt];
         tranfValues = method->GetLayerActivationValues(ev, indexLayer);
         for (itgt = 0; itgt < numOfTargets; itgt++)
            targets[itgt] = ev->GetTarget(itgt);
         R->Fill();
      }
      f->Write();
      f->Close();
      Log() << kINFO << "[AE Transform] New data with transformed variables has been written to " << newDataSetName  << " file" << Endl;
      Double_t regWeight = 1.0;
      TCut myCut = "";
      TFile *transformedData = TFile::Open(newDataSetName);
      TTree *s = (TTree*)transformedData->Get("ae_transformed_regtree");
      transformedLoader->AddRegressionTree(s, regWeight);
      transformedLoader->PrepareTrainingAndTestTree(myCut, fDataSetInfo.GetSplitOptions());
      transformedData->Close();
   }
   else // classification problem
   {
      Log() << kINFO << "[AE Transform] Number of classes: " << numOfClasses << Endl;
      Log() << kINFO << "[AE Transform] Initial number of variables: " << numOfVariables << Endl;
      // create array of trees, each tree represents a class
      const UInt_t N = numOfClasses;
      TTree *classes[N];
      std::vector<Double_t> treeWeights(N);
      for (UInt_t i = 0; i < numOfTranfVariables; i++) {
         varName = "ae_transformed_var";
         varName += i;
         varType = varName;
         varType += "/F";

         Log() << kINFO << "[AE Transform] Adding transformed variable " << varName << " to new DataLoader" << Endl;
         for (UInt_t j = 0; j < numOfClasses; j++) {
            if (i == 0) {// allocate memory to tree pointer
               classes[j] = new TTree(fDataSetInfo.GetClassInfo(j)->GetName(), fDataSetInfo.GetClassInfo(j)->GetName());
            }
            classes[j]->Branch(varName, &tranfValues[i], varType);
         }
         transformedLoader->AddVariable(varName, 'F');
      }

      // loop over all class events, transform and fill the respective trees
      UInt_t itgt, cls;
      for (UInt_t ievt = 0; ievt < nevts; ievt++) {
         ev = events[ievt];
         cls = ev->GetClass();
         treeWeights[cls] = ev->GetOriginalWeight();
         tranfValues = method->GetLayerActivationValues(ev, indexLayer);
         classes[cls]->Fill();
      }
      f->Write();
      f->Close();
      Log() << kINFO << "[AE Transform] New data with transformed variables has been written to " << newDataSetName  << " file" << Endl;
      TFile *transformedData = TFile::Open(newDataSetName);
      for (UInt_t it = 0; it < numOfClasses; it++){
         TTree *s = (TTree*)transformedData->Get(fDataSetInfo.GetClassInfo(it)->GetName());
         transformedLoader->AddTree(s, fDataSetInfo.GetClassInfo(it)->GetName(), treeWeights[it]);
      }
      transformedLoader->PrepareTrainingAndTestTree("", fDataSetInfo.GetSplitOptions());
      transformedData->Close();
   }
   return transformedLoader;
}

////////////////////////////////////////////////////////////////////////////////
/// Feature Clustering

// TMVA::DataLoader* TMVA::VarTransformHandler::FeatureClustering()
// {
// 	const std::vector<Event*>& events = fDataSetInfo.GetDataSet()->GetEventCollection();
// 	TMatrixD* similarityMatrix = GetSimilarityMatrix(events);
// 	return fDataLoader;
// }

// TMatrixD* GetSimilarityMatrix(const std::vector<Event*>& events)
// {
// 	UInt_t nevts = events.size();
//    UInt_t nvars = fDataSetInfo.GetNVariables();
//    Int_t sigma = 1;
// 	TMatrixD* S = new TMatrixD( nevts, nevts );
//    for (UInt_t ievt = 0; ievt < nevts; ievt++)
//    {
//      for (UInt_t jevt = ievt + 1; jevt < nevts; jevt++)
//      {
//          std::vector<Float_t>& valueI = events[ievt]->GetValues();   
//          std::vector<Float_t>& valueJ = events[jevt]->GetValues(); 
//          for (UInt_t k = 0; k < nvars; k++)
//          {
            
//          }
//      }  
//    }

// }

////////////////////////////////////////////////////////////////////////////////
/// Variance Threshold

TMVA::DataLoader* TMVA::VarTransformHandler::VarianceThreshold(Double_t threshold)
{
   CalcNorm();
   const UInt_t nvars = fDataSetInfo.GetNVariables();
   Log() << kINFO << "Number of variables before transformation: " << nvars << Endl;
   std::vector<VariableInfo>& vars = fDataSetInfo.GetVariableInfos();

   // return a new dataloader
   // iterate over all variables, ignore the ones whose variance is below specific threshold
   DataLoader *transformedLoader=(DataLoader *)fDataLoader->Clone(fDataSetInfo.GetName());   
   // TMVA::DataLoader *transformedLoader = new TMVA::DataLoader(fDataSetInfo.GetName());
   Log() << kINFO << "Selecting variables whose variance is above threshold value = " << threshold << Endl;
   Int_t maxL = fDataSetInfo.GetVariableNameMaxLength();
   maxL = maxL + 16;
   Log() << kINFO << "----------------------------------------------------------------" << Endl;
   Log() << kINFO << std::setiosflags(std::ios::left) << std::setw(maxL) << "Selected Variables";
   Log() << kINFO << std::setiosflags(std::ios::left) << std::setw(10) << "Variance" << Endl;
   Log() << kINFO << "----------------------------------------------------------------" << Endl;
   for (UInt_t ivar=0; ivar<nvars; ivar++) {
      Double_t variance =  vars[ivar].GetVariance();
      if (variance > threshold)
      {
         Log() << kINFO << std::setiosflags(std::ios::left) << std::setw(maxL) << vars[ivar].GetExpression();
         Log() << kINFO << std::setiosflags(std::ios::left) << std::setw(maxL) << variance << Endl;
         transformedLoader->AddVariable(vars[ivar].GetExpression(), vars[ivar].GetVarType());
      }
   }
   Log() << kINFO << "----------------------------------------------------------------" << Endl;
   // CopyDataLoader(transformedLoader, fDataLoader);
   // DataLoader *transformedLoader=(DataLoader *)fDataLoader->Clone(fDataSetInfo.GetName());
   transformedLoader->PrepareTrainingAndTestTree(fDataLoader->GetDataSetInfo().GetCut("Signal"), fDataLoader->GetDataSetInfo().GetCut("Background"), fDataLoader->GetDataSetInfo().GetSplitOptions());
   Log() << kINFO << "Number of variables after transformation: " << transformedLoader->GetDataSetInfo().GetNVariables() << Endl;

   return transformedLoader;
}

////////////////////////////////////////////////////////////////////////////////
/// Random Projection

TMVA::DataLoader* TMVA::VarTransformHandler::RandomProjection()
{
	return fDataLoader;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns a single tree containing all the trees of the dataset

TTree* TMVA::VarTransformHandler::MakeDataSetTree() 
{	
   TTree *t = new TTree("Dataset", "Contains all events");
   const std::vector<Event*>& events = fDataSetInfo.GetDataSet()->GetEventCollection();
   std::vector<VariableInfo>& vars = fDataSetInfo.GetVariableInfos();
   UInt_t nvars = fDataSetInfo.GetNVariables();
   UInt_t nevts = events.size();
   TString varName, varType;
   std::vector<Float_t>& values = events[0]->GetValues();
   for (UInt_t i = 0; i < nvars; i++) {
      varName = vars[i].GetExpression();
      varType = varName;
      varType += "/F";
      t->Branch(varName, &values[i], varType);
   }
   for (UInt_t ievt = 0; ievt < nevts; ievt++)
   {
      values = events[ievt]->GetValues();
      t->Fill();
   }
   return t;
}



////////////////////////////////////////////////////////////////////////////////
/// Updates maximum and minimum value of a variable or target

void TMVA::VarTransformHandler::UpdateNorm (Int_t ivar, Double_t x)
{
   Int_t nvars = fDataSetInfo.GetNVariables();
   std::vector<VariableInfo>& vars = fDataSetInfo.GetVariableInfos();
   std::vector<VariableInfo>& tars = fDataSetInfo.GetTargetInfos();
   if( ivar < nvars ){
      if (x < vars[ivar].GetMin()) vars[ivar].SetMin(x);
      if (x > vars[ivar].GetMax()) vars[ivar].SetMax(x);
   }
   else{
      if (x < tars[ivar-nvars].GetMin()) tars[ivar-nvars].SetMin(x);
      if (x > tars[ivar-nvars].GetMax()) tars[ivar-nvars].SetMax(x);
   }
}
	
////////////////////////////////////////////////////////////////////////////////
/// Computes maximum, minimum, mean, RMS and variance for all
/// variables and targets

void TMVA::VarTransformHandler::CalcNorm()
{
   const std::vector<TMVA::Event*>& events = fDataSetInfo.GetDataSet()->GetEventCollection();

   const UInt_t nvars = fDataSetInfo.GetNVariables();
   const UInt_t ntgts = fDataSetInfo.GetNTargets();
   std::vector<VariableInfo>& vars = fDataSetInfo.GetVariableInfos();
   std::vector<VariableInfo>& tars = fDataSetInfo.GetTargetInfos();

   UInt_t nevts = events.size();

   TVectorD x2( nvars+ntgts ); x2 *= 0;
   TVectorD x0( nvars+ntgts ); x0 *= 0;
   TVectorD v0( nvars+ntgts ); v0 *= 0;

   Double_t sumOfWeights = 0;
   for (UInt_t ievt=0; ievt<nevts; ievt++) {
      const Event* ev = events[ievt];

      Double_t weight = ev->GetWeight();
      sumOfWeights += weight;
      for (UInt_t ivar=0; ivar<nvars; ivar++) {
         Double_t x = ev->GetValue(ivar);
         if (ievt==0) {
            vars[ivar].SetMin(x);
            vars[ivar].SetMax(x);
         }
         else {
            UpdateNorm(ivar,  x );
         }
         x0(ivar) += x*weight;
         x2(ivar) += x*x*weight;
      }
      for (UInt_t itgt=0; itgt<ntgts; itgt++) {
         Double_t x = ev->GetTarget(itgt);
         if (ievt==0) {
            tars[itgt].SetMin(x);
            tars[itgt].SetMax(x);
         }
         else {
            UpdateNorm( nvars+itgt,  x );
         }
         x0(nvars+itgt) += x*weight;
         x2(nvars+itgt) += x*x*weight;
      }
   }

   if (sumOfWeights <= 0) {
      Log() << kFATAL << " the sum of event weights calcualted for your input is == 0"
            << " or exactly: " << sumOfWeights << " there is obviously some problem..."<< Endl;
   }

   // set Mean and RMS
   for (UInt_t ivar=0; ivar<nvars; ivar++) {
      Double_t mean = x0(ivar)/sumOfWeights;

      vars[ivar].SetMean( mean );
      if (x2(ivar)/sumOfWeights - mean*mean < 0) {
         Log() << kFATAL << " the RMS of your input variable " << ivar
               << " evaluates to an imaginary number: sqrt("<< x2(ivar)/sumOfWeights - mean*mean
               <<") .. sometimes related to a problem with outliers and negative event weights"
               << Endl;
      }
      vars[ivar].SetRMS( TMath::Sqrt( x2(ivar)/sumOfWeights - mean*mean) );
   }
   for (UInt_t itgt=0; itgt<ntgts; itgt++) {
      Double_t mean = x0(nvars+itgt)/sumOfWeights;
      tars[itgt].SetMean( mean );
      if (x2(nvars+itgt)/sumOfWeights - mean*mean < 0) {
         Log() << kFATAL << " the RMS of your target variable " << itgt
               << " evaluates to an imaginary number: sqrt(" << x2(nvars+itgt)/sumOfWeights - mean*mean
               <<") .. sometimes related to a problem with outliers and negative event weights"
               << Endl;
      }
      tars[itgt].SetRMS( TMath::Sqrt( x2(nvars+itgt)/sumOfWeights - mean*mean) );
   }

   // calculate variance
   for (UInt_t ievt=0; ievt<nevts; ievt++) {
      const Event* ev = events[ievt];
      Double_t weight = ev->GetWeight();

      for (UInt_t ivar=0; ivar<nvars; ivar++) {
         Double_t x = ev->GetValue(ivar);
         Double_t mean = vars[ivar].GetMean();
         v0(ivar) += weight*(x-mean)*(x-mean);
      }

      for (UInt_t itgt=0; itgt<ntgts; itgt++) {
         Double_t x = ev->GetTarget(itgt);
         Double_t mean = tars[itgt].GetMean();
         v0(nvars+itgt) += weight*(x-mean)*(x-mean);
      }
   }

   Int_t maxL = fDataSetInfo.GetVariableNameMaxLength();
   maxL = maxL + 8;
   Log() << kINFO << "----------------------------------------------------------------" << Endl;
   Log() << kINFO << std::setiosflags(std::ios::left) << std::setw(maxL) << "Variables";
   Log() << kINFO << std::setiosflags(std::ios::left) << std::setw(10) << "Variance" << Endl;
   Log() << kINFO << "----------------------------------------------------------------" << Endl;

   // set variance
   Log() << std::setprecision(5);
   for (UInt_t ivar=0; ivar<nvars; ivar++) {
      Double_t variance = v0(ivar)/sumOfWeights;
      vars[ivar].SetVariance( variance );
      Log() << kINFO << std::setiosflags(std::ios::left) << std::setw(maxL) << vars[ivar].GetExpression();
      Log() << kINFO << std::setiosflags(std::ios::left) << std::setw(maxL) << variance << Endl;
   }

   maxL = fDataSetInfo.GetTargetNameMaxLength();
   maxL = maxL + 8;
   Log() << kINFO << "----------------------------------------------------------------" << Endl;
   Log() << kINFO << std::setiosflags(std::ios::left) << std::setw(maxL) << "Targets";
   Log() << kINFO << std::setiosflags(std::ios::left) << std::setw(10) << "Variance" << Endl;
   Log() << kINFO << "----------------------------------------------------------------" << Endl;

   for (UInt_t itgt=0; itgt<ntgts; itgt++) {
      Double_t variance = v0(nvars+itgt)/sumOfWeights;
      tars[itgt].SetVariance( variance );
      Log() << kINFO << std::setiosflags(std::ios::left) << std::setw(maxL) << tars[itgt].GetExpression();
      Log() << kINFO << std::setiosflags(std::ios::left) << std::setw(maxL) << variance << Endl;
   }

   Log() << kINFO << "Set minNorm/maxNorm for variables to: " << Endl;
   Log() << std::setprecision(3);
   for (UInt_t ivar=0; ivar<nvars; ivar++)
      Log() << "    " << vars[ivar].GetExpression()
            << "\t: [" << vars[ivar].GetMin() << "\t, " << vars[ivar].GetMax() << "\t] " << Endl;
   Log() << kINFO << "Set minNorm/maxNorm for targets to: " << Endl;
   Log() << std::setprecision(3);
   for (UInt_t itgt=0; itgt<ntgts; itgt++)
      Log() << "    " << tars[itgt].GetExpression()
            << "\t: [" << tars[itgt].GetMin() << "\t, " << tars[itgt].GetMax() << "\t] " << Endl;
   Log() << std::setprecision(5); // reset to better value	
}

//_______________________________________________________________________
// void TMVA::VarTransformHandler::CopyDataLoader(TMVA::DataLoader* des, TMVA::DataLoader* src)
// {
//    for( std::vector<TreeInfo>::const_iterator treeinfo=src->DataInput().Sbegin();treeinfo!=src->DataInput().Send();treeinfo++)
//    {
//       des->AddSignalTree( (*treeinfo).GetTree(), (*treeinfo).GetWeight(),(*treeinfo).GetTreeType());
//    }

//    for( std::vector<TreeInfo>::const_iterator treeinfo=src->DataInput().Bbegin();treeinfo!=src->DataInput().Bend();treeinfo++)
//    {
//       des->AddBackgroundTree( (*treeinfo).GetTree(), (*treeinfo).GetWeight(),(*treeinfo).GetTreeType());
//    }
// }

