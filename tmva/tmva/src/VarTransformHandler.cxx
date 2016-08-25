#include "TMath.h"
#include "TVectorD.h"
#include "TFile.h"
#include "TTree.h"
#include "TMatrix.h"
#include "TMatrixTSparse.h"
#include "TMatrixDSparsefwd.h"

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
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::VarTransformHandler::VarTransformHandler( DataLoader* dl )
   : fLogger     ( new MsgLogger(TString("VarTransformHandler").Data(), kINFO) ),
     fDataSetInfo(dl->GetDataSetInfo()),
     fDataLoader (dl),
     fEvents (fDataSetInfo.GetDataSet()->GetEventCollection())
{
   Log() << kINFO << "Number of events - " << fEvents.size() << Endl;
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::VarTransformHandler::~VarTransformHandler()
{
	// do something
   delete fLogger;
}


////////////////////////////////////////////////////////////////////////////////
/// Computes variance of all the variables and
/// returns a new DataLoader with the selected variables whose variance is above a specific threshold.
/// Threshold can be provided by user otherwise default value is 0 i.e. remove the variables which have same value in all
/// the events.
///
/// \param[in] threshold value (Double)
///
/// Transformation Definition String Format: "VT(optional float value)"
///
/// Usage examples:
///
/// String    | Description
/// -------   |----------------------------------------
/// "VT"      | Select variables whose variance is above threshold value = 0 (Default)
/// "VT(1.5)" | Select variables whose variance is above threshold value = 1.5

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
/// Autoencoder Transform

TMVA::DataLoader* TMVA::VarTransformHandler::AutoencoderTransform(TString dnnOptions, TString preTrngValue, Int_t indexLayer)
{
   // TODO implement pre-training
   preTrngValue = "false";
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
   method->SetAnalysisType( fAnalysisType );
   method->SetupMethod();
   method->ParseOptions();
   method->ProcessSetup();
   method->CheckSetup();

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
/// Hessian Local Linear Embedding

// TMVA::DataLoader* TMVA::VarTransformHandler::HessianLocalLinearEmbedding(Int_t no_dims, Int_t k)
// {

//    std::vector<VariableInfo>& vars = fDataSetInfo.GetVariableInfos();
//    UInt_t nvars = fDataSetInfo.GetNVariables();
//    UInt_t nevts = fEvents.size();
//    // compute data matrix
//    for (UInt_t ievt=0; ievt<nevts; ievt++) {
//       const Event* ev = fEvents[ievt];
//       // Double_t weight = ev->GetWeight();
//       // sumOfWeights += weight;
//       for (UInt_t ivar=0; ivar<nvars; ivar++) {
//          Double_t x = ev->GetValue(ivar);
//          data(ievt, ivar) = x;
//       }
//    }

//    // Find nearest neighbours
//    std::pair<TMatrixD, TMatrixD> kmap;
//    kmap = FindNearestNeighbours(data, k);
//    Int_t max_k = kmap.second.GetNcols();
//    TMatrixD nind = kmap.second;

//    // Extra term count for quadratic term
//    Int_t dp = no_dims * (no_dims + 1) / 2;
//    TMatrixDSparse W(dp * nevts, nevts);

//    // For all datapoints
//    Log() << kINFO << "Building Hessian Estimator for neighbouring points" << Endl;
//    for (i = 0 ; i < nevts ; i++)
//    {
//       // Center datapoints by subtracting their mean
//       TMatrixD thisx(k, nvars);
//       for (UInt_t ii = 0; ii < k; ii++)
//       {
//          for (UInt_t jj = 0; jj < nvars; jj++)
//             thisx(ii, jj) = data(nind(i, ii), jj);
//       }
//       thisx = thisx - RepeatMatrix(GetColsMean(thisx), k, 1);
//       TMatrix Vpr = ComputeSVDMatrix(thisx);
//       if (Vpr.GetNcols() < no_dims)
//       {
//          no_dims = Vpr.GetNcols();
//          dp = no_dims * (no_dims + 1)/2;
//          Log() << kWarning << "Number of dimensions are being reduced to " << no_dims << Endl;
//       }
//       V = Vpr.SliceMatrix(0,0, Vpr.GetNrows() - 1, no_dims - 1);

//       // Get Hessian estimator
//       TMatrixD Yi = GetHessianEstimator(V, no_dims);
//       std::pair<TMatrix, TMatrix> Y = GramSchOrthogonalisation(Yi);
//       TMatrix Pii = Y.first.SliceMatrix(0, Y.first.GetNrows() - 1, no_dims + 1, Y.first.GetNcols() - 1).Transpose();
//       for (j = 0; j < dp; j++)
//       {
//          if (Pii.SliceMatrix(j, 0, 0, Pii.GetNcols()).Sum() > 0.0001)
//             tpp = Pii.SliceMatrix(j, 0, 0, Pii.GetNcols()) / Pii.SliceMatrix(j, 0, 0, Pii.GetNcols()).Sum();
//          else
//             tpp = Pii.SliceMatrix(j, 0, 0, Pii.GetNcols());
//          // correct it
//          W.AssignVector(tpp, );
//       }
//    }

//    Log() << kINFO << "Computing Hessian LLE embedding" << Endl;
//    TMatrixDSparse G = W.Transpose() * W;

//    // TODO clear memory
//    TMatrixD eigenVectors = G.EigenVectors();
//    newData = eigenVectors.SliceMatrix(0, eigenVectors.GetNrows(), 1, no_dims + 1);
//    newData = newData * sqrt(n);

//    // return a new dataloader with transformed variables
//    DataLoader *transformedLoader=(DataLoader *)fDataLoader->Clone(fDataSetInfo.GetName());

//    return fDataLoader;
// }

///////////////////////////////////////////////////////////////////////////////
////////////////////////////// Utility methods ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// Repeats the matrix

TMatrixD& TMVA::VarTransformHandler::RepeatMatrix(TMatrixD& mat, Int_t x, Int_t y)
{
   Int_t n = mat.GetNrows();
   Int_t d = mat.GetNcols();
   TMatrixD* repeat_mat = new TMatrixD(n*x, y*d);
   for (Int_t itr = 0; itr < x; itr++)
   {
      for (Int_t itc = 0; itc < y; itc++)
      {
         for (Int_t i = 0; i < n; i++)
         {
            for (Int_t j = 0; j < d; j++)
            {
               (*repeat_mat)(itr*n + i, itc*d + j) = mat(i, j);
            }
         }
      }
   }
   return *repeat_mat;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns mean of each column of a matrix

TMatrixD& TMVA::VarTransformHandler::GetColsMean(TMatrixD& mat)
{
   Int_t n = mat.GetNrows();
   Int_t d = mat.GetNcols();
   Double_t sum = 0;
   TMatrixD* col_mean = new TMatrixD(1, d);
   for (Int_t i = 0; i < d; i++)
   {
      sum = 0;
      for (Int_t j = 0; j < n; j++)
      {
         sum += mat(j, i);
      }
      Log() << kINFO << sum << Endl;
      (*col_mean)(0, i) = sum/n;
   }
   return (*col_mean);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns mean of each column of a matrix

TMatrixD& TMVA::VarTransformHandler::GetRowsMean(TMatrixD& mat)
{
   Int_t n = mat.GetNrows();
   Int_t d = mat.GetNcols();
   Double_t sum = 0;
   TMatrixD* row_mean = new TMatrixD(n, 1);
   for (Int_t i = 0; i < n; i++)
   {
      sum = 0;
      for (Int_t j = 0; j < d; j++)
      {
         sum += mat(i, j);
      }
      Log() << kINFO << sum << Endl;
      (*row_mean)(i, 0) = sum/n;
   }
   return (*row_mean);
}

////////////////////////////////////////////////////////////////////////////////
/// Slice the matrix

TMatrixD& TMVA::VarTransformHandler::SliceMatrix(TMatrixD& mat, Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb)
{
   if (row_lwb > row_upb)
   {
      Log() << kINFO << "Lower bound of row number for slicing the matrix is greater than" <<
                        "upper bound, please recheck. Returning original matrix as of now" << Endl;
      return mat;
   }
   else if (col_lwb > col_upb)
   {
      Log() << kINFO << "Lower bound of column number for slicing the matrix is greater than" <<
                        "upper bound, please recheck. Returning original matrix as of now" << Endl;
      return mat;
   }
   TMatrixD* sliced_matrix = new TMatrixD(row_upb - row_lwb + 1, col_upb - col_lwb + 1);
   for (Int_t i = row_lwb; i <= row_upb; i++)
   {
      for (Int_t j = col_lwb; j <= col_upb; j++)
      {
         (*sliced_matrix)(i - row_lwb, j - col_lwb) = mat(i, j);
      }
   }
   return (*sliced_matrix);
}

////////////////////////////////////////////////////////////////////////////////
/// Calculates 2 norm or Euclidean Norm of a matrix

Double_t TMVA::VarTransformHandler::GetMatrixNorm(TMatrixD& mat)
{
   Int_t n = mat.GetNrows();
   Int_t d = mat.GetNcols();
   Double_t sum = 0;
   for (Int_t i = 0; i < n; i++)
   {
      for (Int_t j = 0; j < d; j++)
         sum += (mat(i, j)*mat(i, j));
   }
   return sqrt(sum);
}

// ////////////////////////////////////////////////////////////////////////////////
// /// Performs Gram Schmidt Orthogonalization of a matrix

std::pair<TMatrixD, TMatrixD> TMVA::VarTransformHandler::GramSchOrthogonalisation(TMatrixD& mat)
{
   Int_t nrows = mat.GetNrows();
   Int_t ncols = mat.GetNcols();
   TMatrixD V(nrows, ncols);
   V = mat;
   TMatrixD R(ncols, ncols);

   std::pair<TMatrixD, TMatrixD> Q(V,R);
   for (Int_t i = 0; i < ncols; i++)
   {
      R(i, i) = GetMatrixNorm(SliceMatrix(V, 0, nrows-1, i, i));
      for (Int_t it = 0; it < nrows; it++)
         V(it, i) = V(it, i)/R(i, i);
      if (i < ncols-1)
      {
         for (Int_t j = i + 1; j < ncols; j++)
         {
            TMatrixD X = SliceMatrix(V, 0, nrows-1, i, i);
            X.Transpose(X);
            TMatrixD Y = SliceMatrix(V, 0, nrows-1, j, j);
            TMatrixD Z(1,1);
            Z.Mult(X,Y);
            R(i, j) = Z(0,0);
            for (Int_t k = 0; k < nrows; k++)
            {
               V(k, j) = V(k, j) - R(i, j) * V(k, i);
            }
         }
      }
   }
   Q.first = V;
   Q.second = R;
   return Q;
}

// ////////////////////////////////////////////////////////////////////////////////
// /// TODO UDV' = SVD(X), Computes SVD of a matrix and  returns V

// TMatrixD& TMVA::VarTransformHandler::ComputeSVDMatrix(TMatrixD& mat)
// {

// }

// ////////////////////////////////////////////////////////////////////////////////
// /// Returns the Hessian estimate of a matrix

// TMatrixD& TMVA::VarTransformHandler::GetHessianEstimator(TMatrixD& mat, Int_t no_dims)
// {

//    // TODO Compute appropiate size of Yi
//    TMatrixD Yi(nrows, );
//    Int_t ct = 0;
//    Int_t nrows = mat.GetNrows();
//    Int_t ncols = mat.GetNcols();
//    for (Int_t i = 0; i < no_dims; i++)
//    {
//       TMatrixD startp = SliceMatrix(mat, 0, nrows, i, i);
//       for (Int_t j = 0; j < (no_dims - i + 1); j++)
//       {
//          for (Int_t k = 0; k < nrows; k++)
//          {
//             Yi(k, ct + j) = startp(k, 0) * V(k, i + j);
//          }
//       }
//       ct = ct + no_dims - i + 1;
//    }
//    TMatrixD Y(V.GetNrows(), V.GetNcols() + 1 + Yi.GetNcols())
//    for (Int_t i = 0; i < V.GetNrows(); i++)
//       Y(i, 0) = 1;
//    for (Int_t i = 0; i < V.GetNrows(); i++)
//    {
//       for (Int_t j = 0; j < V.GetNcols(); j++)
//          Y(i, j + 1) = V(i, j);
//    }
//    Int_t V_rows = V.GetNrows();
//    Int_t V_cols = V.GetNcols();
//    for (Int_t i = 0; i < Yi.GetNrows(); i++)
//    {
//       for (Int_t j = 0; j < Yi.GetNcols(); j++)
//          Y(i + v_rows, j + 1 + v_cols) = Yi(i, j);
//    }
//    return Y;
// }

// // TODO: modify it by event weight
std::pair<TMatrixD, TMatrixD> TMVA::VarTransformHandler::FindNearestNeighbours(TMatrixD& data, Int_t k)
{
   UInt_t nevts = data.GetNrows();
   UInt_t nvars = data.GetNcols();
   Log() << kINFO << nevts << " - number of events" << Endl;
   Log() << kINFO << nvars << " - number of variables" << Endl;


   // find distance between events and pick k nearest neighbour for each event
   TMatrixD D(nevts, k); D *= 0;
   TMatrixD ni(nevts, k); ni *= 0;
   std::vector< std::pair<Double_t, Int_t> > distances(nevts);
   std::pair<TMatrixD, TMatrixD> kmap(D, ni);
   for (UInt_t i = 0; i < nevts; i++)
   {
      for (UInt_t j = 0; j < nevts; j++)
      {
         Double_t d = 0;
         for (UInt_t ivar = 0; ivar < nvars; ivar++)
         {
            d += ((data(i, ivar) - data(j, ivar)) * (data(i, ivar) - data(j, ivar)));
         }
         distances[j].first = sqrt(d);
         distances[j].second = j;
      }
      std::sort(distances.begin(), distances.end());
      for (Int_t it = 1; it < k + 1; it++) {
         D(i, it-1) = distances[it].first;
         ni(i, it-1) = distances[it].second;
         // Log() << kINFO << "(" << i << "," << it << ") " << D(i, it-1) << Endl;
         // Log() << kINFO << ni(i, it - 1) << " ";
      }
      Log() << kINFO << Endl;
   }
   kmap.first = D;
   kmap.second = ni;
   return kmap;
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

