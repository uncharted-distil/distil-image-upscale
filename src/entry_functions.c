#include "entry_functions.h"
#include "utility.c"
#include <stdio.h>
#define NUM_SUPPORTED_MODELS 2
// globals
Model* model[NUM_SUPPORTED_MODELS];
ModelInfo* modelInfo[NUM_SUPPORTED_MODELS];

void initialize(char* errorMsg){
    // create model
    model[NoiseCancel] = newModel();
    // get NoiseCancel model info (currently only supported model)
    modelInfo[NoiseCancel] = &supportedModels[NoiseCancel];
    TFInfo check = loadModel(model[NoiseCancel], modelInfo[NoiseCancel]->directoryLocation, modelInfo[NoiseCancel]->tag);
    if(check.code)
    {
        freeModel(model[NoiseCancel]); // free memory from model
        strcpy(errorMsg, TF_Message(check.status));
        return;
    }
    check = findModelNodes(model[NoiseCancel], modelInfo[NoiseCancel]);
    if(check.code)
    {
        freeModel(model[NoiseCancel]); // free memory from model
        strcpy(errorMsg, TF_Message(check.status));
        return;
    }
    // create GAN model
    model[GAN] = newModel();
    // get NoiseCancel model info (currently only supported model)
    modelInfo[GAN] = &supportedModels[GAN];
    check = loadModel(model[GAN], modelInfo[GAN]->directoryLocation, modelInfo[GAN]->tag);
    if(check.code)
    {
        freeModel(model[GAN]); // free memory from model
        strcpy(errorMsg, TF_Message(check.status));
        return;
    }
    check = findModelNodes(model[GAN], modelInfo[GAN]);
    if(check.code)
    {
        freeModel(model[GAN]); // free memory from model
        strcpy(errorMsg, TF_Message(check.status));
        return;
    }
}
// on success call freeOutputData to avoid memory leak, returns null and errorMsg if errors
OutputData* runModel(char* errorMsg, ModelTypes modelType, DataInfo dataInfo){
    // create input and output tensor on heap
    TF_Tensor* inputTensor[modelInfo[modelType]->numInputNodes]; 
    TF_Tensor* outputTensor[1] = {NULL};
    // create tensors for all the static data required for models
    for(unsigned int i=0; i < modelInfo[modelType]->numStaticInputData; ++i){
        TFInfo check = dataInfoToTensor(inputTensor, &modelInfo[modelType]->staticInputData[i], model[modelType]->status, i);
        if(check.code){
            strcpy(errorMsg, TF_Message(check.status));
            freeTensor(inputTensor, modelInfo[modelType]->numInputNodes);
            free(outputTensor[0]);
            return NULL;
        }
    }

    // create tensors from the passed in data
    TFInfo check = dataInfoToTensor(inputTensor, &dataInfo, model[modelType]->status, modelInfo[modelType]->numStaticInputData);
    if(check.code){
        strcpy(errorMsg, TF_Message(check.status));
        freeTensor(inputTensor, modelInfo[modelType]->numInputNodes);
        free(outputTensor[0]);
        return NULL;
    }
    // run model on the tensors
    check = run(model[modelType], modelInfo[modelType], inputTensor, outputTensor);
    // delete input tensors no longer needed
    freeTensor(inputTensor, modelInfo[modelType]->numInputNodes);
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
    freeModel(model[NoiseCancel]);
    freeModel(model[GAN]);
}
