#include "models.c"
#include "macros.c"
#include "utility.c"
#include <stdio.h>

// globals
Model* model = NULL;
ModelInfo* modelInfo = NULL;

EXPORTED void initialize(const char* errorMsg){
    // create model
    model = newModel();
    // get NoiseCancel model info (currently only supported model)
    modelInfo = &supportedModels[NoiseCancel];
    TFInfo check = loadModel(model, modelInfo->directoryLocation, modelInfo->tag);
    if(check.code)
    {
        freeModel(model); // free memory from model
        errorMsg = TF_Message(check.status);
        return;
    }
    check = findModelNodes(model, modelInfo);
    if(check.code)
    {
        freeModel(model); // free memory from model
        errorMsg = TF_Message(check.status);
        return;
    }
}
// on success call freeOutputData to avoid memory leak, returns null and errorMsg if errors
EXPORTED OutputData* runModel(const char* errorMsg, DataInfo dataInfo){
    // create input and output tensor on heap
    TF_Tensor** inputTensor=(TF_Tensor**)malloc(sizeof(TF_Tensor*) * modelInfo->numInputNodes);
    TF_Tensor* outputTensor = malloc(sizeof(TF_Tensor*));
    // create tensors for all the static data required for models
    for(unsigned int i=0; i < modelInfo->numStaticInputData; ++i){
        TFInfo check = dataInfoToTensor(&inputTensor, &modelInfo->staticInputData[i], model->status, i);
        if(check.code){
            errorMsg = TF_Message(check.status);
            freeTensor(inputTensor, modelInfo->numInputNodes);
            free(outputTensor);
            return NULL;
        }
    }
    // create tensors from the passed in data
    TFInfo check = dataInfoToTensor(&inputTensor, &dataInfo, model->status, modelInfo->numStaticInputData);
    if(check.code){
        errorMsg = TF_Message(check.status);
        freeTensor(inputTensor, modelInfo->numInputNodes);
        free(outputTensor);
        return NULL;
    }
    // run model on the tensors
    check = run(model, modelInfo, inputTensor, &outputTensor);
    if(check.code){
        errorMsg = TF_Message(check.status);
        freeTensor(inputTensor, modelInfo->numInputNodes);
        freeTensor(&outputTensor, 1);
        return NULL;
    }
    // create outputdata to hold all the information required
    OutputData *outputData = (OutputData *)malloc(sizeof(outputData));
    outputData->buffer = (float *)TF_TensorData(outputTensor);
    outputData->outputTensor = outputTensor;
    outputData->numOfDimensions = TF_NumDims(outputTensor);
    outputData->dimension = (float *)malloc(sizeof(float) * outputData->numOfDimensions);
    for(unsigned int i=0; i < outputData->numOfDimensions; ++i){
        outputData->dimension[i] = TF_Dim(outputTensor, i);
    }
    return outputData;
}

EXPORTED void freeOutputData(OutputData *outputData){
    freeTensor(&outputData->outputTensor, 1);
    free(outputData->dimension);
    free(outputData);
}

EXPORTED void cleanup(){
    freeModel(model);
}