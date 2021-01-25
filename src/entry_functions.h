#include "models.c"
#include "macros.c"

EXPORTED OutputData* runModel(const char* errorMsg, DataInfo dataInfo);
EXPORTED void initialize(const char* errorMsg);
EXPORTED void freeOutputData(OutputData *outputData);
EXPORTED void cleanup();