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

      VarTransformHandler(DataSetInfo&, DataLoader*);
      ~VarTransformHandler();

      TMVA::DataLoader* VarianceThreshold(Double_t threshold);
      TMVA::DataLoader* AutoencoderTransform(TString dnnOptions, TString preTrngValue, Int_t indexLayer);
      TMVA::DataLoader* FeatureClustering();
      TMVA::DataLoader* RandomProjection();  

      mutable MsgLogger* fLogger;             //! message logger
      MsgLogger& Log() const { return *fLogger; }

   private:

      DataSetInfo&             fDataSetInfo; 
      DataLoader*              fDataLoader;
      TTree*                   MakeDataSetTree();
      void                     UpdateNorm (Int_t ivar, Double_t x); 
      void                     CalcNorm();
      // void                     CopyDataLoader(DataLoader* des, DataLoader* src);      

   };

}