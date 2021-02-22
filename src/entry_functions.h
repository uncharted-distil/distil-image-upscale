#include "models.c"
#include "macros.c"

EXPORTED OutputData* runModel(char* errorMsg, ModelTypes model, DataInfo dataInfo);
EXPORTED void initialize(char* errorMsg, ModelTypes type);
EXPORTED void freeOutputData(OutputData *outputData);
EXPORTED void cleanup(ModelTypes type);