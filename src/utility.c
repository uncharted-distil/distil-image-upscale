#pragma once
#include "type_defines.c"
#include <stdio.h>
/******************DEBUG UTILITY*****************/
// logger
TF_Code logStatus(TFInfo info, const char* taskMessage) {
    if(info.code == TF_OK){
        printf("%s: OK \n", taskMessage);
        return info.code;
    }
    printf("TASK:%s: %s\n", taskMessage, TF_Message(info.status));
    return info.code;
}

// helper function for debugging
TFInfo newTFInfo(TF_Status* status, TF_Code code, const char* msg){
        TF_SetStatus(status, code, msg);
        return (TFInfo){code:code, status:status};
}
/******************Data & Model Runtime*****************/
// noOperation function to disable tensorflow from deleting data memory, this responsibility is left to us
void NoOpDeallocator(void* data, size_t a, void* b) {
    free(data);
}

// loads model from directory
TFInfo loadModel(Model* model, const char* modelDirectory, const char* tags){
    int ntags = 1;
    model->session = TF_LoadSessionFromSavedModel(model->sessionOptions, model->runOptions,
     modelDirectory, &tags, ntags, model->graph, NULL, model->status);
    return (TFInfo){code:TF_GetCode(model->status), status:model->status};  
}

TFInfo dataInfoToTensor(TF_Tensor** inputTensor, DataInfo* dataInfo, TF_Status* status, unsigned int index){
    inputTensor[index] = TF_AllocateTensor(dataInfo->dataType, dataInfo->dimensions, dataInfo->numberOfDimensions, dataInfo->dataSize);
    memcpy(TF_TensorData(inputTensor[index]), dataInfo->data, dataInfo->dataSize);
    if(inputTensor[index] == NULL){
        char buffer[256];
        sprintf(buffer, "Input tensor could not be created. numDims: %i, dataSize: %i, dims:[%i,%i,%i,%i]", 
        (int)dataInfo->numberOfDimensions, 
        (int)dataInfo->dataSize,
        (int)dataInfo->dimensions[0],
        (int)dataInfo->dimensions[1],
        (int)dataInfo->dimensions[2],
        (int)dataInfo->dimensions[3]);
        return newTFInfo(status, TF_ABORTED, buffer);
    }
    return newTFInfo(status, TF_OK, "");
}

TFInfo run(Model* model, ModelInfo* modelInfo,TF_Tensor** tensor, TF_Tensor** outputTensor){
    TF_SessionRun(model->session, 
    NULL, model->input, 
    tensor, modelInfo->numInputNodes, 
    model->output, 
    outputTensor, 1, // assumes 1 output tensor
    NULL, 0, NULL, model->status);

return(TFInfo){code:TF_GetCode(model->status), status: model->status};
}
/******************Model Management*****************/
// create and initialize model to default values must call freeModel when done with returned instance
Model* newModel(){
    Model *model = malloc(sizeof(Model));
    model->graph = TF_NewGraph();
    model->status = TF_NewStatus();
    model->sessionOptions = TF_NewSessionOptions();
    model->session = NULL;
    model->runOptions = NULL;
    return model;
}; 

// free all memory associated to model
void freeModel(Model* model){
    TF_DeleteGraph(model->graph);
    TF_DeleteSession(model->session, model->status);
    TF_DeleteSessionOptions(model->sessionOptions);
    TF_DeleteStatus(model->status);
    free(model->output);
    free(model->input);
    free(model);
}
void freeTensor(TF_Tensor** tensors, unsigned int numTensors)
{
    for(unsigned int i=0; i < numTensors; ++i)
    {
        TF_DeleteTensor(tensors[i]);
    }
}
/******************Model Information*****************/
// outputs all the graph's operation names
void printGraph(TF_Graph* graph){
    size_t position=0;
    TF_Operation *currentOperation;
    while((currentOperation = TF_GraphNextOperation(graph, &position)) != NULL)
    {
        ++position;
        printf("%s \n", TF_OperationName(currentOperation));
    }
}

// finds all model nodes required to run (input and output nodes) should only call this function once upon loading the model
TFInfo findModelNodes(Model* model, ModelInfo* modelInfo){
    if(model->graph == NULL)
    {
        return newTFInfo(model->status, TF_FAILED_PRECONDITION, "graph must not be null");
    }
    // allocate array to hold all pointers to input nodes for model
    model->input = malloc(sizeof(TF_Output)*modelInfo->numInputNodes);
    for(unsigned int i=0; i < modelInfo->numInputNodes; ++i)
    {
        model->input[i] = (TF_Output){TF_GraphOperationByName(model->graph, modelInfo->inputNodeNames[i]), 0};
        if(model->input[i].oper == NULL){
        return newTFInfo(model->status, TF_ABORTED, "could not find input node");
        }
    }
    // assumes 1 output node
    model->output = malloc(sizeof(TF_Output));
    model->output[0] = (TF_Output){TF_GraphOperationByName(model->graph, modelInfo->outputNodeName), 0};
    if(model->output[0].oper == NULL)
    {
       return newTFInfo(model->status, TF_ABORTED, "could not find output node"); 
    }
    return (TFInfo){code:TF_OK, status:NULL};
}