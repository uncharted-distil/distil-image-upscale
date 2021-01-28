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
    TF_Tensor* inputTensor[modelInfo->numInputNodes]; //=malloc(sizeof(TF_Tensor*) * modelInfo->numInputNodes);
    TF_Tensor* outputTensor[1] = {NULL};
    // create tensors for all the static data required for models
    for(unsigned int i=0; i < modelInfo->numStaticInputData; ++i){
        TFInfo check = dataInfoToTensor(inputTensor, &modelInfo->staticInputData[i], model->status, i);
        if(check.code){
            strcpy(errorMsg, TF_Message(check.status));
            freeTensor(inputTensor, modelInfo->numInputNodes);
            free(outputTensor[0]);
            return NULL;
        }
    }

    // create tensors from the passed in data
    TFInfo check = dataInfoToTensor(inputTensor, &dataInfo, model->status, modelInfo->numStaticInputData);
    if(check.code){
        strcpy(errorMsg, TF_Message(check.status));
        freeTensor(inputTensor, modelInfo->numInputNodes);
        free(outputTensor[0]);
        return NULL;
    }
    // run model on the tensors
    check = run(model, modelInfo, inputTensor, outputTensor);
    // delete input tensors no longer needed
    freeTensor(inputTensor, modelInfo->numInputNodes);
    if(check.code){
        strcpy(errorMsg, TF_Message(check.status));
        TF_DeleteTensor(outputTensor[0]);
        return NULL;
    }
    // create outputdata to hold all the information required
    OutputData *outputData = malloc(sizeof(outputData));
    // allocate memory to hold new data
    outputData->buffer = malloc(TF_TensorByteSize(outputTensor[0]));
    // copy memory over from output tensor
    memcpy(outputData->buffer, TF_TensorData(outputTensor[0]), TF_TensorByteSize(outputTensor[0]));
    // create dimension meta information
    outputData->numOfDimensions = TF_NumDims(outputTensor[0]);
    outputData->dimension = malloc(sizeof(int64_t) * outputData->numOfDimensions);
    for(unsigned int i=0; i < outputData->numOfDimensions; ++i){
        outputData->dimension[i] = TF_Dim(outputTensor[0], i);
    }
    // delete output tensor no longer needed
    TF_DeleteTensor(outputTensor[0]);
    return outputData;
}

void freeOutputData(OutputData *outputData){
    free(outputData->buffer);
    outputData->buffer=NULL;
    free(outputData->dimension);
    outputData->dimension=NULL;
    free(outputData);
}

void cleanup(){
    freeModel(model);
}
