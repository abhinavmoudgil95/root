#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_DataSetInfo
#include "TMVA/DataSetInfo.h"
#endif

class TTree;
class TFile;
class TDirectory;

namespace TMVA {

   class DataLoader;
   class MethodBase;
   class MethodDNN;
   class DataSetInfo;
   class Event;
   class DataSet;
   class MsgLogger;
   class DataInputHandler;
   class VarTransformHandler {
   public:

      VarTransformHandler(DataLoader*);
      ~VarTransformHandler();

      TMVA::DataLoader* VarianceThreshold(Double_t threshold);
      TMVA::DataLoader* DeepAutoencoder(TString dnnOptions, TString preTrngValue, Int_t indexLayer);
      TMVA::DataLoader* FeatureClustering();
      TMVA::DataLoader* HessianLocalLinearEmbedding(Int_t no_dims, Int_t k);
      mutable MsgLogger* fLogger;             //! message logger
      MsgLogger& Log() const { return *fLogger; }

   private:

      DataSetInfo&                  fDataSetInfo;
      DataLoader*                   fDataLoader;
      const std::vector<Event*>&    fEvents;
      TTree*                        MakeDataSetTree();
      void                          UpdateNorm (Int_t ivar, Double_t x);
      void                          CalcNorm();
      TMatrixD&                     RepeatMatrix(TMatrixD& mat, Int_t x, Int_t y);
      TMatrixD&                     SliceMatrix(TMatrixD& mat, Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb);
      TMatrixD&                     GetColsMean(TMatrixD& mat);
      TMatrixD&                     GetRowsMean(TMatrixD& mat);
      std::pair<TMatrixD, TMatrixD> FindNearestNeighbours(TMatrixD& data, Int_t k);
      Double_t                      GetMatrixNorm(TMatrixD& mat);
      std::pair<TMatrixD, TMatrixD> GramSchOrthogonalisation(TMatrixD& mat);
      void                          CopyDataLoader(TMVA::DataLoader* des, TMVA::DataLoader* src);
   };

}
