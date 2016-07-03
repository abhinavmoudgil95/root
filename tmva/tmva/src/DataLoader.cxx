// @(#)root/tmva $Id$   
// Author: Omar Zapata
// Mentors: Lorenzo Moneta, Sergei Gleyzer
//NOTE: Based on TMVA::Factory

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : DataLoader                                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      This is a class to load datasets into every booked method                 *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Lorenzo Moneta <Lorenzo.Moneta@cern.ch> - CERN, Switzerland               *
 *      Omar Zapata <Omar.Zapata@cern.ch>  - ITM/UdeA, Colombia                   *
 *      Sergei Gleyzer<sergei.gleyzer@cern.ch> - CERN, Switzerland                *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      ITM/UdeA, Colombia                                                        *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <iomanip>
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TEventList.h"
#include "TH2.h"
#include "TText.h"
#include "TStyle.h"
#include "TMatrixF.h"
#include "TMatrixDSym.h"
#include "TPaletteAxis.h"
#include "TPrincipal.h"
#include "TMath.h"
#include "TObjString.h"
#include "TRandom3.h"
#include "TSystem.h"

#include "TMVA/DataLoader.h"
#include "TMVA/Config.h"
#include "TMVA/Tools.h"
#include "TMVA/Ranking.h"
#include "TMVA/DataSet.h"
#include "TMVA/IMethod.h"
#include "TMVA/MethodBase.h"
#include "TMVA/MethodDNN.h"
#include "TMVA/DataInputHandler.h"
#include "TMVA/DataSetManager.h"
#include "TMVA/DataSetInfo.h"
#include "TMVA/MethodBoost.h"
#include "TMVA/MethodCategory.h"
#include "TMVA/ClassifierFactory.h" 

#include "TMVA/VariableIdentityTransform.h"
#include "TMVA/VariableDecorrTransform.h"
#include "TMVA/VariablePCATransform.h"
#include "TMVA/VariableGaussTransform.h"
#include "TMVA/VariableNormalizeTransform.h"

#include "TMVA/ResultsClassification.h"
#include "TMVA/ResultsRegression.h"
#include "TMVA/ResultsMulticlass.h"

#include "TMVA/Types.h"
//const Int_t  MinNoTrainingEvents = 10;
//const Int_t  MinNoTestEvents     = 1;

ClassImp(TMVA::DataLoader)


//_______________________________________________________________________
TMVA::DataLoader::DataLoader( TString thedlName)
: Configurable( ),
   fDataSetManager       ( NULL ), //DSMTEST
   fDataInputHandler     ( new DataInputHandler ),
   fTransformations      ( "I" ),
   fVerbose              ( kFALSE ),
   fName                 ( thedlName ),
   fCalcNorm             ( kFALSE ),   
   fDataAssignType       ( kAssignEvents ),
   fATreeEvent           ( NULL )
{

   //   DataSetManager::CreateInstance(*fDataInputHandler); // DSMTEST removed
   fDataSetManager = new DataSetManager( *fDataInputHandler ); // DSMTEST

   // render silent
   //    if (gTools().CheckForSilentOption( GetOptions() )) Log().InhibitOutput(); // make sure is silent if wanted to
}


//_______________________________________________________________________
TMVA::DataLoader::~DataLoader( void )
{
   // destructor
   //   delete fATreeEvent;

   std::vector<TMVA::VariableTransformBase*>::iterator trfIt = fDefaultTrfs.begin();
   for (;trfIt != fDefaultTrfs.end(); trfIt++) delete (*trfIt);

   delete fDataInputHandler;

   // destroy singletons
   //   DataSetManager::DestroyInstance(); // DSMTEST replaced by following line
   delete fDataSetManager; // DSMTEST

   // problem with call of REGISTER_METHOD macro ...
   //   ClassifierDataLoader::DestroyInstance();
   //   Types::DestroyInstance();
   Tools::DestroyInstance();
   Config::DestroyInstance();
}


//_______________________________________________________________________
TMVA::DataSetInfo& TMVA::DataLoader::AddDataSet( DataSetInfo &dsi )
{
   return fDataSetManager->AddDataSetInfo(dsi); // DSMTEST
}

//_______________________________________________________________________
TMVA::DataSetInfo& TMVA::DataLoader::AddDataSet( const TString& dsiName )
{
   DataSetInfo* dsi = fDataSetManager->GetDataSetInfo(dsiName); // DSMTEST

   if (dsi!=0) return *dsi;
   
   return fDataSetManager->AddDataSetInfo(*(new DataSetInfo(dsiName))); // DSMTEST
}

//_______________________________________________________________________
TMVA::DataSetInfo& TMVA::DataLoader::GetDataSetInfo()
{
   return DefaultDataSetInfo(); // DSMTEST
}

////////////////////////////////////////////////////////////////////////////////
/// Updates maximum and minimum value of a variable or target

void TMVA::DataLoader::UpdateNorm ( Int_t ivar,  Double_t x ) 
{
   Int_t nvars = DefaultDataSetInfo().GetNVariables();
   std::vector<VariableInfo>& vars = DefaultDataSetInfo().GetVariableInfos();   
   std::vector<VariableInfo>& tars = DefaultDataSetInfo().GetTargetInfos();      
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

void TMVA::DataLoader::CalcNorm() 
{
   fCalcNorm = kTRUE;
   const std::vector<Event*>& events = DefaultDataSetInfo().GetDataSet()->GetEventCollection();

   const UInt_t nvars = DefaultDataSetInfo().GetNVariables();
   const UInt_t ntgts = DefaultDataSetInfo().GetNTargets();
   std::vector<VariableInfo>& vars = DefaultDataSetInfo().GetVariableInfos();   
   std::vector<VariableInfo>& tars = DefaultDataSetInfo().GetTargetInfos();    

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

   Int_t maxL = DefaultDataSetInfo().GetVariableNameMaxLength();
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

   maxL = DefaultDataSetInfo().GetTargetNameMaxLength();
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
void TMVA::DataLoader::CopyDataLoader(TMVA::DataLoader* des, TMVA::DataLoader* src)
{
   for( std::vector<TreeInfo>::const_iterator treeinfo=src->DataInput().Sbegin();treeinfo!=src->DataInput().Send();treeinfo++)
   {
      des->AddSignalTree( (*treeinfo).GetTree(), (*treeinfo).GetWeight(),(*treeinfo).GetTreeType());
   }

   for( std::vector<TreeInfo>::const_iterator treeinfo=src->DataInput().Bbegin();treeinfo!=src->DataInput().Bend();treeinfo++)
   {
      des->AddBackgroundTree( (*treeinfo).GetTree(), (*treeinfo).GetWeight(),(*treeinfo).GetTreeType());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Transform the events by Autoencoder and return a new DataLoader

TMVA::DataLoader* TMVA::DataLoader::AETransform(MethodDNN *method, const std::vector<Event*>& events, Int_t indexLayer)
{
	// get number of variables for new DataLoader
	TMVA::DataLoader *transformedLoader = new TMVA::DataLoader(DefaultDataSetInfo().GetName());
    const Event* ev = events[0];
	std::vector<Float_t>& tranfValues = method->GetLayerActivationValues(ev, indexLayer);
	Int_t numOfTranfVariables = tranfValues.size();

	// create a new dataset file 
    TString newDataSetName = DefaultDataSetInfo().GetName();
    newDataSetName += "_ae_transformed.root";
	TFile *f = new TFile(newDataSetName,"RECREATE");

	// get number of classes
	UInt_t numOfClasses = DefaultDataSetInfo().GetNClasses();
	UInt_t numOfTargets = DefaultDataSetInfo().GetNTargets();
	UInt_t numOfVariables = DefaultDataSetInfo().GetNVariables();
	UInt_t nevts = events.size();

	// it's a regression problem
	if (numOfTargets != 0)
	{
		// create a new tree with transformed variables and original targets
		TString varName, tarName, varType, tarType;
		TTree *R = new TTree("R","AE Transformed Regression Tree");
		for (Int_t i = 0; i < numOfTranfVariables; i++) {
			varName = "aeTransformedVar";
			varName += i;
			varType = varName;
			varType += "/F";
			R->Branch(varName, &tranfValues[i], varType);
			transformedLoader->AddVariable(varName, 'F');
		}
		std::vector<VariableInfo>& tars = DefaultDataSetInfo().GetTargetInfos();  
		std::vector<Float_t> targets(numOfTargets);  
		for (UInt_t i = 0; i < numOfTargets; i++) {
			tarType = tars[i].GetExpression();
			tarType += "/F";
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
		Double_t regWeight = 1.0;
		TCut myCut = "";
		transformedLoader->AddRegressionTree(R, regWeight);
		transformedLoader->PrepareTrainingAndTestTree(myCut, DefaultDataSetInfo().GetSplitOptions());
	}
	else // classification problem
	{

	}
	return transformedLoader;

}

////////////////////////////////////////////////////////////////////////////////
/// Computes variance of all the variables and 
/// returns a new DataLoader with the selected variables whose variance is above a specific threshold. 
/// Threshold can be provided by user otherwise default value is 0 i.e. remove the variables which have same value in all 
/// the events. 
/// 
/// \param[in] trafoDefinition Tranformation Definition String
///
/// Transformation Definition String Format: "VT(optional float value)"
/// 
/// Usage examples: 
/// 
/// String    | Description
/// -------   |----------------------------------------
/// "VT"      | Select variables whose variance is above threshold value = 0 (Default)
/// "VT(1.5)" | Select variables whose variance is above threshold value = 1.5

TMVA::DataLoader* TMVA::DataLoader::VarTransform(TString trafoDefinition)
{
   TString trOptions = "0";
   TString trName = "None";
   if (trafoDefinition.Contains("(")) { 

      // contains transformation parameters
      Ssiz_t parStart = trafoDefinition.Index( "(" );
      Ssiz_t parLen   = trafoDefinition.Index( ")", parStart )-parStart+1;

      trName = trafoDefinition(0,parStart);
      trOptions = trafoDefinition(parStart,parLen);
      trOptions.Remove(parLen-1,1);
      trOptions.Remove(0,1);       
   }
   else
      trName = trafoDefinition;

   // variance threshold variable transformation
   if (trName == "VT") {

      // find threshold value from given input
      Double_t threshold = 0.0;
      if (!trOptions.IsFloat()){
         Log() << kFATAL << " VT transformation must be passed a floating threshold value" << Endl; 
         return this;
      }
      else
         threshold =  trOptions.Atof();      
      Log() << kINFO << "Transformation: " << trName << Endl; 
      Log() << kINFO << "Threshold value: " << threshold << Endl;

      // calculate variance of variables if not done already
      if(!fCalcNorm)
         CalcNorm();

      // get variable info
      const UInt_t nvars = DefaultDataSetInfo().GetNVariables();
      Log() << kINFO << "Number of variables before transformation: " << nvars << Endl; 
      std::vector<VariableInfo>& vars = DefaultDataSetInfo().GetVariableInfos();   

      // return a new dataloader
      // iterate over all variables, ignore the ones whose variance is below specific threshold 
      TMVA::DataLoader *transformedloader = new TMVA::DataLoader(DefaultDataSetInfo().GetName());
      Log() << kINFO << "Selecting variables whose variance is above threshold value = " << threshold << Endl;  
   	  Int_t maxL = DefaultDataSetInfo().GetVariableNameMaxLength();
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
            transformedloader->AddVariable(vars[ivar].GetExpression(), vars[ivar].GetVarType());
         }
      }  
      Log() << kINFO << "----------------------------------------------------------------" << Endl; 
      CopyDataLoader(transformedloader,this);
      transformedloader->PrepareTrainingAndTestTree(this->DefaultDataSetInfo().GetCut("Signal"), this->DefaultDataSetInfo().GetCut("Background"), this->DefaultDataSetInfo().GetSplitOptions());
      Log() << kINFO << "Number of variables after transformation: " << transformedloader->DefaultDataSetInfo().GetNVariables() << Endl;

      return transformedloader;
   }
   // Autoencoder Variable Transformation
   else if (trName == "AE") {

   		// add targets to the current loader 
   		std::vector<VariableInfo>& vars = DefaultDataSetInfo().GetVariableInfos();  
   		const UInt_t nvars = DefaultDataSetInfo().GetNVariables();
		for (UInt_t ivar=0; ivar<nvars; ivar++) {
			this->AddTarget(vars[ivar].GetExpression());
		}

		// CombineTrees(this);
		// extract names contained in "trOptions"
		Types::EMVA theMethod = TMVA::Types::kDNN;
		TString theMethodName = Types::Instance().GetMethodName( theMethod );
		TString datasetname = DefaultDataSetInfo().GetName();
		TString JobName = "TMVAAnalysis";

   		/// Book DNN Method 
	  	Event::SetIsTraining(kTRUE);  	
		gSystem->MakeDirectory(this->GetName());
		fAnalysisType = Types::kRegression;
		TString methodTitle = "DNN";
		IMethod* im;		
		im = ClassifierFactory::Instance().Create( std::string(theMethodName),
	                                         JobName,
	                                         methodTitle,
	                                         DefaultDataSetInfo(),
	                                         trOptions );
		MethodDNN *method = dynamic_cast<MethodDNN*>(im);
		if (method==0)
		{
			Log() << kINFO << "------------------------method = 0----------------------------" << Endl; 
			return this;
		} 
		if (fAnalysisType == Types::kRegression) {
 			Log() << "Regression with " << DefaultDataSetInfo().GetNTargets() << " targets." << Endl;
		} 
		method->SetAnalysisType( fAnalysisType );
		method->SetupMethod();
		method->ParseOptions();
		method->ProcessSetup();
		method->CheckSetup();

		// Train DNN Method
		Log() << kINFO << "Train method: " << method->GetMethodName() << " for Regression" << Endl;
        method->TrainMethod();		
        Log() << kINFO << "Training finished" << Endl;

        Int_t indexLayer = 1;
        const std::vector<Event*>& events = DefaultDataSetInfo().GetDataSet()->GetEventCollection();
        TMVA::DataLoader* transformedLoader = AETransform(method, events, indexLayer);
        // return transformedLoader; 

  //       FetchOriginalTrees(transformedloader);
  //       // Loop over all events and compute new features 
  //       // Create a new root file with new TTree containing signal and background tree

  //       // Initiate a new root file
  //       TString newDataSetName = DefaultDataSetInfo().GetName();
  //       newDataSetName += "_transformed.root";
  //       Int_t indexLayer = 1;

  //       // create a new TTree
  //       TTree *S = new TTree("S","AE Transformed Signal Tree");
  //       TTree *B = new TTree("B","AE Transformed Background Tree");
  //       TFile *f = new TFile(newDataSetName,"RECREATE");
  //       const std::vector<Event*>& events = DefaultDataSetInfo().GetDataSet()->GetEventCollection();
  //       UInt_t nevts = events.size();
  //       TMVA::DataLoader *transformedloader = new TMVA::DataLoader(newDataSetName);

  //       // get number of variables in hidden layer
  //       const Event* ev = events[0];
		// std::vector<Float_t>& actValuesSig = method->GetLayerActivationValues(ev, indexLayer);
		// std::vector<Float_t>& actValuesBkg = method->GetLayerActivationValues(ev, indexLayer);
		// Int_t numOfVariables = actValuesSig.size();

		// // create a new tree with transformed variables 
		// // Float_t varArr[numOfVariables];
		// TString varName;
		// TString varType;
		// for (Int_t i = 0; i < numOfVariables; i++) {
		// 	varName = "var";
		// 	varName += i;
		// 	varType = varName;
		// 	varType += "/F";
		// 	S->Branch(varName, &actValuesSig[i], varType);
		// 	B->Branch(varName, &actValuesBkg[i], varType);
		// 	transformedloader->AddVariable(varName, 'F');
		// }

		// // fill the tree with activation values of hidden layers i.e. transformed variable values
		// // std::vector<double>& actValues = method->GetLayerActivationValues(ev, indexLayer);
		// UInt_t cls;
		// for (UInt_t ievt=1; ievt<nevts; ievt++) {
		// 	ev = events[ievt];
		// 	if (DefaultDataSetInfo().IsSignal(ev)){
		// 		actValuesSig = method->GetLayerActivationValues(ev, indexLayer);
		// 		S->Fill();
		// 	}
		// 	else {
		// 		actValuesBkg = method->GetLayerActivationValues(ev, indexLayer);
		// 		B->Fill();
		// 	}
		// }        
		// f->Write();
		// transformedloader->AddSignalTree(S, 1.0);
		// transformedloader->AddBackgroundTree(B, 1.0);
		// transformedloader->PrepareTrainingAndTestTree(this->DefaultDataSetInfo().GetCut("Signal"), this->DefaultDataSetInfo().GetCut("Background"), this->DefaultDataSetInfo().GetSplitOptions());
		// Log() << kINFO << "Number of variables after transformation: " << transformedloader->DefaultDataSetInfo().GetNVariables() << Endl;
		// return transformedloader;
   }
   else {
      Log() << kFATAL << "Incorrect transformation string provided, please check" << Endl;
   }
   return this;
}

// ________________________________________________
// the next functions are to assign events directly 

//_______________________________________________________________________
TTree* TMVA::DataLoader::CreateEventAssignTrees( const TString& name )
{
   // create the data assignment tree (for event-wise data assignment by user)
   TTree * assignTree = new TTree( name, name );
   assignTree->SetDirectory(0);
   assignTree->Branch( "type",   &fATreeType,   "ATreeType/I" );
   assignTree->Branch( "weight", &fATreeWeight, "ATreeWeight/F" );

   std::vector<VariableInfo>& vars = DefaultDataSetInfo().GetVariableInfos();
   std::vector<VariableInfo>& tgts = DefaultDataSetInfo().GetTargetInfos();
   std::vector<VariableInfo>& spec = DefaultDataSetInfo().GetSpectatorInfos();

   if (!fATreeEvent) fATreeEvent = new Float_t[vars.size()+tgts.size()+spec.size()];
   // add variables
   for (UInt_t ivar=0; ivar<vars.size(); ivar++) {
      TString vname = vars[ivar].GetExpression();
      assignTree->Branch( vname, &(fATreeEvent[ivar]), vname + "/F" );
   }
   // add targets
   for (UInt_t itgt=0; itgt<tgts.size(); itgt++) {
      TString vname = tgts[itgt].GetExpression();
      assignTree->Branch( vname, &(fATreeEvent[vars.size()+itgt]), vname + "/F" );
   }
   // add spectators
   for (UInt_t ispc=0; ispc<spec.size(); ispc++) {
      TString vname = spec[ispc].GetExpression();
      assignTree->Branch( vname, &(fATreeEvent[vars.size()+tgts.size()+ispc]), vname + "/F" );
   }
   return assignTree;
}

//_______________________________________________________________________
void TMVA::DataLoader::AddSignalTrainingEvent( const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal training event
   AddEvent( "Signal", Types::kTraining, event, weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddSignalTestEvent( const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal testing event
   AddEvent( "Signal", Types::kTesting, event, weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddBackgroundTrainingEvent( const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal training event
   AddEvent( "Background", Types::kTraining, event, weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddBackgroundTestEvent( const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal training event
   AddEvent( "Background", Types::kTesting, event, weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddTrainingEvent( const TString& className, const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal training event
   AddEvent( className, Types::kTraining, event, weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddTestEvent( const TString& className, const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal test event
   AddEvent( className, Types::kTesting, event, weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddEvent( const TString& className, Types::ETreeType tt,
                                 const std::vector<Double_t>& event, Double_t weight ) 
{
   // add event
   // vector event : the order of values is: variables + targets + spectators
   ClassInfo* theClass = DefaultDataSetInfo().AddClass(className); // returns class (creates it if necessary)
   UInt_t clIndex = theClass->GetNumber();


   // set analysistype to "kMulticlass" if more than two classes and analysistype == kNoAnalysisType
   if( fAnalysisType == Types::kNoAnalysisType && DefaultDataSetInfo().GetNClasses() > 2 )
      fAnalysisType = Types::kMulticlass;

   
   if (clIndex>=fTrainAssignTree.size()) {
      fTrainAssignTree.resize(clIndex+1, 0);
      fTestAssignTree.resize(clIndex+1, 0);
   }

   if (fTrainAssignTree[clIndex]==0) { // does not exist yet
      fTrainAssignTree[clIndex] = CreateEventAssignTrees( Form("TrainAssignTree_%s", className.Data()) );
      fTestAssignTree[clIndex]  = CreateEventAssignTrees( Form("TestAssignTree_%s",  className.Data()) );
   }
   
   fATreeType   = clIndex;
   fATreeWeight = weight;
   for (UInt_t ivar=0; ivar<event.size(); ivar++) fATreeEvent[ivar] = event[ivar];

   if(tt==Types::kTraining) fTrainAssignTree[clIndex]->Fill();
   else                     fTestAssignTree[clIndex]->Fill();

}

//_______________________________________________________________________
Bool_t TMVA::DataLoader::UserAssignEvents(UInt_t clIndex) 
{
   // 
   return fTrainAssignTree[clIndex]!=0;
}

//_______________________________________________________________________
void TMVA::DataLoader::SetInputTreesFromEventAssignTrees()
{
   // assign event-wise local trees to data set
   UInt_t size = fTrainAssignTree.size();
   for(UInt_t i=0; i<size; i++) {
      if(!UserAssignEvents(i)) continue;
      const TString& className = DefaultDataSetInfo().GetClassInfo(i)->GetName();
      SetWeightExpression( "weight", className );
      AddTree(fTrainAssignTree[i], className, 1.0, TCut(""), Types::kTraining );
      AddTree(fTestAssignTree[i], className, 1.0, TCut(""), Types::kTesting );
   }
}

//_______________________________________________________________________
void TMVA::DataLoader::AddTree( TTree* tree, const TString& className, Double_t weight, 
                                const TCut& cut, const TString& treetype )
{
   // number of signal events (used to compute significance)
   Types::ETreeType tt = Types::kMaxTreeType;
   TString tmpTreeType = treetype; tmpTreeType.ToLower();
   if      (tmpTreeType.Contains( "train" ) && tmpTreeType.Contains( "test" )) tt = Types::kMaxTreeType;
   else if (tmpTreeType.Contains( "train" ))                                   tt = Types::kTraining;
   else if (tmpTreeType.Contains( "test" ))                                    tt = Types::kTesting;
   else {
      Log() << kFATAL << "<AddTree> cannot interpret tree type: \"" << treetype 
            << "\" should be \"Training\" or \"Test\" or \"Training and Testing\"" << Endl;
   }
   AddTree( tree, className, weight, cut, tt );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddTree( TTree* tree, const TString& className, Double_t weight, 
                                const TCut& cut, Types::ETreeType tt )
{
   if(!tree)
      Log() << kFATAL << "Tree does not exist (empty pointer)." << Endl;

   DefaultDataSetInfo().AddClass( className );

   // set analysistype to "kMulticlass" if more than two classes and analysistype == kNoAnalysisType
   if( fAnalysisType == Types::kNoAnalysisType && DefaultDataSetInfo().GetNClasses() > 2 )
      fAnalysisType = Types::kMulticlass;

   Log() << kINFO << "Add Tree " << tree->GetName() << " of type " << className 
         << " with " << tree->GetEntries() << " events" << Endl;
   DataInput().AddTree( tree, className, weight, cut, tt );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddSignalTree( TTree* signal, Double_t weight, Types::ETreeType treetype )
{
   // number of signal events (used to compute significance)
   AddTree( signal, "Signal", weight, TCut(""), treetype );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddSignalTree( TString datFileS, Double_t weight, Types::ETreeType treetype )
{
   // add signal tree from text file

   // create trees from these ascii files
   TTree* signalTree = new TTree( "TreeS", "Tree (S)" );
   signalTree->ReadFile( datFileS );
 
   Log() << kINFO << "Create TTree objects from ASCII input files ... \n- Signal file    : \""
         << datFileS << Endl;
  
   // number of signal events (used to compute significance)
   AddTree( signalTree, "Signal", weight, TCut(""), treetype );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddSignalTree( TTree* signal, Double_t weight, const TString& treetype )
{
   AddTree( signal, "Signal", weight, TCut(""), treetype );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddBackgroundTree( TTree* signal, Double_t weight, Types::ETreeType treetype )
{
   // number of signal events (used to compute significance)
   AddTree( signal, "Background", weight, TCut(""), treetype );
}
//_______________________________________________________________________
void TMVA::DataLoader::AddBackgroundTree( TString datFileB, Double_t weight, Types::ETreeType treetype )
{
   // add background tree from text file

   // create trees from these ascii files
   TTree* bkgTree = new TTree( "TreeB", "Tree (B)" );
   bkgTree->ReadFile( datFileB );
 
   Log() << kINFO << "Create TTree objects from ASCII input files ... \n- Background file    : \""
         << datFileB << Endl;
  
   // number of signal events (used to compute significance)
   AddTree( bkgTree, "Background", weight, TCut(""), treetype );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddBackgroundTree( TTree* signal, Double_t weight, const TString& treetype )
{
   AddTree( signal, "Background", weight, TCut(""), treetype );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetSignalTree( TTree* tree, Double_t weight )
{
   AddTree( tree, "Signal", weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetBackgroundTree( TTree* tree, Double_t weight )
{
   AddTree( tree, "Background", weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetTree( TTree* tree, const TString& className, Double_t weight )
{
   // set background tree
   AddTree( tree, className, weight, TCut(""), Types::kMaxTreeType );
}

//_______________________________________________________________________
void  TMVA::DataLoader::SetInputTrees( TTree* signal, TTree* background, 
                                       Double_t signalWeight, Double_t backgroundWeight )
{
   // define the input trees for signal and background; no cuts are applied
   AddTree( signal,     "Signal",     signalWeight,     TCut(""), Types::kMaxTreeType );
   AddTree( background, "Background", backgroundWeight, TCut(""), Types::kMaxTreeType );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetInputTrees( const TString& datFileS, const TString& datFileB, 
                                      Double_t signalWeight, Double_t backgroundWeight )
{
   DataInput().AddTree( datFileS, "Signal", signalWeight );
   DataInput().AddTree( datFileB, "Background", backgroundWeight );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetInputTrees( TTree* inputTree, const TCut& SigCut, const TCut& BgCut )
{
   // define the input trees for signal and background from single input tree,
   // containing both signal and background events distinguished by the type 
   // identifiers: SigCut and BgCut
   AddTree( inputTree, "Signal",     1.0, SigCut, Types::kMaxTreeType );
   AddTree( inputTree, "Background", 1.0, BgCut , Types::kMaxTreeType );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddVariable( const TString& expression, const TString& title, const TString& unit, 
                                    char type, Double_t min, Double_t max )
{
   // user inserts discriminating variable in data set info
   DefaultDataSetInfo().AddVariable( expression, title, unit, min, max, type ); 
}

//_______________________________________________________________________
void TMVA::DataLoader::AddVariable( const TString& expression, char type,
                                    Double_t min, Double_t max )
{
   // user inserts discriminating variable in data set info
   DefaultDataSetInfo().AddVariable( expression, "", "", min, max, type ); 
}

//_______________________________________________________________________
void TMVA::DataLoader::AddTarget( const TString& expression, const TString& title, const TString& unit, 
                                  Double_t min, Double_t max )
{
   // user inserts target in data set info

   if( fAnalysisType == Types::kNoAnalysisType )
      fAnalysisType = Types::kRegression;

   DefaultDataSetInfo().AddTarget( expression, title, unit, min, max ); 
}

//_______________________________________________________________________
void TMVA::DataLoader::AddSpectator( const TString& expression, const TString& title, const TString& unit, 
                                     Double_t min, Double_t max )
{
   // user inserts target in data set info
   DefaultDataSetInfo().AddSpectator( expression, title, unit, min, max ); 
}

//_______________________________________________________________________
TMVA::DataSetInfo& TMVA::DataLoader::DefaultDataSetInfo() 
{ 
   // default creation
   return AddDataSet( fName );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetInputVariables( std::vector<TString>* theVariables ) 
{ 
   // fill input variables in data set
   for (std::vector<TString>::iterator it=theVariables->begin();
        it!=theVariables->end(); it++) AddVariable(*it);
}

//_______________________________________________________________________
void TMVA::DataLoader::SetSignalWeightExpression( const TString& variable)  
{ 
   DefaultDataSetInfo().SetWeightExpression(variable, "Signal"); 
}

//_______________________________________________________________________
void TMVA::DataLoader::SetBackgroundWeightExpression( const TString& variable) 
{
   DefaultDataSetInfo().SetWeightExpression(variable, "Background");
}

//_______________________________________________________________________
void TMVA::DataLoader::SetWeightExpression( const TString& variable, const TString& className )  
{
   //Log() << kWarning << DefaultDataSetInfo().GetNClasses() /*fClasses.size()*/ << Endl;
   if (className=="") {
      SetSignalWeightExpression(variable);
      SetBackgroundWeightExpression(variable);
   } 
   else  DefaultDataSetInfo().SetWeightExpression( variable, className );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetCut( const TString& cut, const TString& className ) {
   SetCut( TCut(cut), className );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetCut( const TCut& cut, const TString& className ) 
{
   DefaultDataSetInfo().SetCut( cut, className );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddCut( const TString& cut, const TString& className ) 
{
   AddCut( TCut(cut), className );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddCut( const TCut& cut, const TString& className ) 
{
   DefaultDataSetInfo().AddCut( cut, className );
}

//_______________________________________________________________________
void TMVA::DataLoader::PrepareTrainingAndTestTree( const TCut& cut, 
                                                   Int_t NsigTrain, Int_t NbkgTrain, Int_t NsigTest, Int_t NbkgTest,
                                                   const TString& otherOpt )
{
   // prepare the training and test trees
   SetInputTreesFromEventAssignTrees();

   AddCut( cut  );

   DefaultDataSetInfo().SetSplitOptions( Form("nTrain_Signal=%i:nTrain_Background=%i:nTest_Signal=%i:nTest_Background=%i:%s", 
                                              NsigTrain, NbkgTrain, NsigTest, NbkgTest, otherOpt.Data()) );
}

//_______________________________________________________________________
void TMVA::DataLoader::PrepareTrainingAndTestTree( const TCut& cut, Int_t Ntrain, Int_t Ntest )
{
   // prepare the training and test trees 
   // kept for backward compatibility
   SetInputTreesFromEventAssignTrees();

   AddCut( cut  );

   DefaultDataSetInfo().SetSplitOptions( Form("nTrain_Signal=%i:nTrain_Background=%i:nTest_Signal=%i:nTest_Background=%i:SplitMode=Random:EqualTrainSample:!V", 
                                              Ntrain, Ntrain, Ntest, Ntest) );
}

//_______________________________________________________________________
void TMVA::DataLoader::PrepareTrainingAndTestTree( const TCut& cut, const TString& opt )
{
   // prepare the training and test trees
   // -> same cuts for signal and background
   SetInputTreesFromEventAssignTrees();

   DefaultDataSetInfo().PrintClasses();
   AddCut( cut );
   DefaultDataSetInfo().SetSplitOptions( opt );
}

//_______________________________________________________________________
void TMVA::DataLoader::PrepareTrainingAndTestTree( TCut sigcut, TCut bkgcut, const TString& splitOpt )
{
   // prepare the training and test trees

   // if event-wise data assignment, add local trees to dataset first
   SetInputTreesFromEventAssignTrees();

   Log() << kINFO << "Preparing trees for training and testing..." << Endl;
   AddCut( sigcut, "Signal"  );
   AddCut( bkgcut, "Background" );

   DefaultDataSetInfo().SetSplitOptions( splitOpt );
}

//______________________________________________________________________
void TMVA::DataLoader::PrepareTrainingAndTestTree(int foldNumber, Types::ETreeType tt)
{
  DataInput().ClearSignalTreeList();
  DataInput().ClearBackgroundTreeList();

  TString CrossValidate = "ParameterOpt";

  int numFolds = fTrainSigTree.size();

  for(int i=0; i<numFolds; ++i){
    if(CrossValidate == "PerformanceEst"){
      if(i!=foldNumber){
	AddTree( fTrainSigTree.at(i),     "Signal",     1.0,     TCut(""), Types::kTraining );
	AddTree( fTrainBkgTree.at(i),     "Background", 1.0,     TCut(""), Types::kTraining );
	AddTree( fTestSigTree.at(i),      "Signal",     1.0,     TCut(""), Types::kTraining );
	AddTree( fTestBkgTree.at(i),      "Background", 1.0,     TCut(""), Types::kTraining );
      }
      else{
	AddTree( fTrainSigTree.at(i),     "Signal",     1.0,     TCut(""), Types::kTesting );
	AddTree( fTrainBkgTree.at(i),     "Background", 1.0,     TCut(""), Types::kTesting );
	AddTree( fTestSigTree.at(i),      "Signal",     1.0,     TCut(""), Types::kTesting );
	AddTree( fTestBkgTree.at(i),      "Background", 1.0,     TCut(""), Types::kTesting );
      }
    }
    else if(CrossValidate == "ParameterOpt"){
      if(tt == Types::kTraining){
	if(i!=foldNumber){
	  AddTree( fTrainSigTree.at(i),     "Signal",     1.0,     TCut(""), Types::kTraining );
	  AddTree( fTrainBkgTree.at(i),     "Background", 1.0,     TCut(""), Types::kTraining );
	}
	else{
	  AddTree( fTrainSigTree.at(i),     "Signal",     1.0,     TCut(""), Types::kTesting );
	  AddTree( fTrainBkgTree.at(i),     "Background", 1.0,     TCut(""), Types::kTesting );
	}
      }
      else if(tt == Types::kTesting){
	if(i!=foldNumber){
	  AddTree( fTestSigTree.at(i),     "Signal",     1.0,     TCut(""), Types::kTraining );
	  AddTree( fTestBkgTree.at(i),     "Background", 1.0,     TCut(""), Types::kTraining );
	}
	else{
	  AddTree( fTestSigTree.at(i),     "Signal",     1.0,     TCut(""), Types::kTesting );
	  AddTree( fTestBkgTree.at(i),     "Background", 1.0,     TCut(""), Types::kTesting );
	}
      }
    }
  }

}

void TMVA::DataLoader::MakeKFoldDataSet(int numberFolds)
{
  
  UInt_t nSigTrees = DataInput().GetNSignalTrees();
  UInt_t nBkgTrees = DataInput().GetNBackgroundTrees();

  if(nSigTrees == 1){
    std::vector<TTree*> tempSigTrees = SplitSets(DataInput().SignalTreeInfo(0).GetTree(), 1, 2);
    fTrainSigTree = SplitSets(tempSigTrees.at(0), 0, numberFolds);
    fTestSigTree = SplitSets(tempSigTrees.at(1), 1, numberFolds);
  }
  if(nBkgTrees == 1){
    std::vector<TTree*> tempBkgTrees = SplitSets(DataInput().BackgroundTreeInfo(0).GetTree(), 1, 2);
    fTrainBkgTree = SplitSets(tempBkgTrees.at(0), 0, numberFolds);
    fTestBkgTree = SplitSets(tempBkgTrees.at(1), 1, numberFolds);
  }

  for(UInt_t i=0; i<nSigTrees; ++i){
    if(DataInput().SignalTreeInfo(i).GetTreeType() == Types::kTraining){
      fTrainSigTree = SplitSets(DataInput().SignalTreeInfo(i).GetTree(), i, numberFolds);
    }
    else if(DataInput().SignalTreeInfo(i).GetTreeType() == Types::kTesting){
      fTestSigTree = SplitSets(DataInput().SignalTreeInfo(i).GetTree(), i, numberFolds);
    }
  }
  for(UInt_t j=0; j<nBkgTrees; ++j){
    if(DataInput().BackgroundTreeInfo(j).GetTreeType() == Types::kTraining){
      fTrainBkgTree = SplitSets(DataInput().BackgroundTreeInfo(j).GetTree(), j, numberFolds);
    }
    else if(DataInput().BackgroundTreeInfo(j).GetTreeType() == Types::kTesting){
      fTestBkgTree = SplitSets(DataInput().BackgroundTreeInfo(j).GetTree(), j, numberFolds);
    }
  }

  DataInput().ClearSignalTreeList();
  DataInput().ClearBackgroundTreeList();

  nSigTrees = DataInput().GetNSignalTrees();
  nBkgTrees = DataInput().GetNBackgroundTrees();

}

void TMVA::DataLoader::ValidationKFoldSet(){
  DefaultDataSetInfo().GetDataSet()->DivideTrainingSet(2);
  DefaultDataSetInfo().GetDataSet()->MoveTrainingBlock(1, Types::kValidation, kTRUE);
}

std::vector<TTree*> TMVA::DataLoader::SplitSets(TTree * oldTree, int seedNum, int numFolds)
{
  std::vector<TTree*> tempTrees;

  for(int l=0; l<numFolds; ++l){
    tempTrees.push_back(oldTree->CloneTree(0));
    tempTrees.at(l)->SetDirectory(0);
  }

  TRandom3 r(seedNum);

  Long64_t nEntries = oldTree->GetEntries();

  std::vector<TBranch*> branches;

  //TBranch * typeBranch = oldTree->GetBranch("type");
  //branches.push_back(typeBranch);
  //oldTree->SetBranchAddress( "type",   &fATreeType);
  //TBranch * weightBranch = oldTree->GetBranch("weight");
  //branches.push_back(weightBranch);
  //oldTree->SetBranchAddress( "weight", &fATreeWeight);

  std::vector<VariableInfo>& vars = DefaultDataSetInfo().GetVariableInfos();
  std::vector<VariableInfo>& tgts = DefaultDataSetInfo().GetTargetInfos();
  std::vector<VariableInfo>& spec = DefaultDataSetInfo().GetSpectatorInfos();

  UInt_t varsSize = vars.size();

  if (!fATreeEvent) fATreeEvent = new Float_t[vars.size()+tgts.size()+spec.size()];
  // add variables
  for (UInt_t ivar=0; ivar<vars.size(); ivar++) {
    TString vname = vars[ivar].GetExpression();
    if(vars[ivar].GetExpression() != vars[ivar].GetLabel()){
      varsSize--;
      continue; 
    }
    TBranch * branch = oldTree->GetBranch(vname);
    branches.push_back(branch);
    oldTree->SetBranchAddress(vname, &(fATreeEvent[ivar]));
  }
  // add targets
  for (UInt_t itgt=0; itgt<tgts.size(); itgt++) {
    TString vname = tgts[itgt].GetExpression();
    if(tgts[itgt].GetExpression() != tgts[itgt].GetLabel()){ continue; }
    TBranch * branch = oldTree->GetBranch(vname);
    branches.push_back(branch);
    oldTree->SetBranchAddress( vname, &(fATreeEvent[vars.size()+itgt]));
  }
  // add spectators
  for (UInt_t ispc=0; ispc<spec.size(); ispc++) {
    TString vname = spec[ispc].GetExpression();
    if(spec[ispc].GetExpression() != spec[ispc].GetLabel()){ continue; }
    TBranch * branch = oldTree->GetBranch(vname);
    branches.push_back(branch);
    oldTree->SetBranchAddress( vname, &(fATreeEvent[vars.size()+tgts.size()+ispc]));
  }

  Long64_t foldSize = nEntries/numFolds;
  Long64_t inSet = 0;

  for(Long64_t i=0; i<nEntries; i++){
    for(UInt_t j=0; j<vars.size(); j++){ fATreeEvent[j]=0; }
    oldTree->GetEvent(i);
    bool inTree = false;
    if(inSet == foldSize*numFolds){
      break;
    }
    else{
      while(!inTree){
	int s = r.Integer(numFolds);
	if(tempTrees.at(s)->GetEntries()<foldSize){
	  tempTrees.at(s)->Fill();
	  inSet++;
	  inTree=true;
	}
      }
    }
  }

  return tempTrees;
}
