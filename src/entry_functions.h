#include "models.c"
#include "macros.c"

EXPORTED OutputData* runModel(char* errorMsg, DataInfo dataInfo);
EXPORTED void initialize(char* errorMsg);
EXPORTED void freeOutputData(OutputData *outputData);
EXPORTED void cleanup();