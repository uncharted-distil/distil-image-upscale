#include "entry_functions.h"
#include "utility.c"
#include <stdio.h>

// globals
Model* model = NULL;
ModelInfo* modelInfo = NULL;

void initialize(char* errorMsg){
    // create model
    model = newModel();
    // get NoiseCancel model info (currently only supported model)
    modelInfo = &supportedModels[NoiseCancel];
    TFInfo check = loadModel(model, modelInfo->directoryLocation, modelInfo->tag);
    if(check.code)
    {
        freeModel(model); // free memory from model
        strcpy(errorMsg, TF_Message(check.status));
        return;
    }
    check = findModelNodes(model, modelInfo);
    if(check.code)
    {
        freeModel(model); // free memory from model
        strcpy(errorMsg, TF_Message(check.status));
        return;
    }
}
// on success call freeOutputData to avoid memory leak, returns null and errorMsg if errors
OutputData* runModel(char* errorMsg, DataInfo dataInfo){
    // create input and output tensor on heap
    TF_Tensor** inputTensor=(TF_Tensor**)malloc(sizeof(TF_Tensor*) * modelInfo->numInputNodes);
    TF_Tensor* outputTensor=malloc(sizeof(TF_Tensor*));
    // create tensors for all the static data required for models
    for(unsigned int i=0; i < modelInfo->numStaticInputData; ++i){
        TFInfo check = dataInfoToTensor(&inputTensor, &modelInfo->staticInputData[i], model->status, i);
        if(check.code){
            strcpy(errorMsg, TF_Message(check.status));
            freeTensor(inputTensor, modelInfo->numInputNodes);
            free(outputTensor);
            return NULL;
        }
    }
    // create tensors from the passed in data
    TFInfo check = dataInfoToTensor(&inputTensor, &dataInfo, model->status, modelInfo->numStaticInputData);
    if(check.code){
        strcpy(errorMsg, TF_Message(check.status));
        freeTensor(inputTensor, modelInfo->numInputNodes);
        free(outputTensor);
        return NULL;
    }
    // run model on the tensors
    check = run(model, modelInfo, inputTensor, &outputTensor);
    if(check.code){
        strcpy(errorMsg, TF_Message(check.status));
        freeTensor(inputTensor, modelInfo->numInputNodes);
        freeTensor(&outputTensor, 1);
        return NULL;
    }
    // create outputdata to hold all the information required
    OutputData *outputData = (OutputData *)malloc(sizeof(outputData));
    outputData->buffer = (float *)TF_TensorData(outputTensor);
    outputData->outputTensor = outputTensor;
    outputData->numOfDimensions = TF_NumDims(outputTensor);
    outputData->dimension = (int64_t *)malloc(sizeof(int64_t) * outputData->numOfDimensions);
    for(unsigned int i=0; i < outputData->numOfDimensions; ++i){
        outputData->dimension[i] = TF_Dim(outputTensor, i);
    }
    return outputData;
}

void freeOutputData(OutputData *outputData){
    freeTensor(&outputData->outputTensor, 1);
    free(outputData->dimension);
    free(outputData);
}

void cleanup(){
    freeModel(model);
}
