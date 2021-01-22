#include "type_defines.c"
#include <stdio.h>

enum ModelTypes{
    NoiseCancel,
};
ModelInfo supportedModels[1] = {{directoryLocation:"./models/noise_cancel", inputNodeName: "serving_default_LR", outputNodeName:"StatefulPartitionedCall", tag:"serve"}};
void NoOpDeallocator(void* data, size_t a, void* b) {}
TF_Code logStatus(TFInfo info, const char* taskMessage) {
    if(info.code == TF_OK){
        printf("%s: OK \n", taskMessage);
        return info.code;
    }
    printf("TASK:%s: %s\n", taskMessage, TF_Message(info.status));
    return info.code;
}
Model* newModel(){
    Model *model = malloc(sizeof(Model));
    model->graph = TF_NewGraph();
    model->status = TF_NewStatus();
    model->sessionOptions = TF_NewSessionOptions();
    model->session = NULL;
    model->runOptions = NULL;
    return model;
}; 
TFInfo loadModel(Model* model, const char* modelDirectory, const char* tags){
    int ntags = 1;
    model->session = TF_LoadSessionFromSavedModel(model->sessionOptions, model->runOptions,
     modelDirectory, &tags, ntags, model->graph, NULL, model->status);
    return (TFInfo){code:TF_GetCode(model->status), status:model->status};  
}
TFInfo newTFInfo(TF_Status* status, TF_Code code, const char* msg){
        TF_SetStatus(status, code, msg);
        return (TFInfo){code:code, status:status};
}
TFInfo findModelNodes(Model* model, ModelInfo* modelInfo){
    if(model->graph == NULL)
    {
        return newTFInfo(model->status, TF_FAILED_PRECONDITION, "graph must not be null");
    }

    // assumes 1 input node
    model->input = malloc(sizeof(TF_Input));
    *model->input = (TF_Output){TF_GraphOperationByName(model->graph, modelInfo->inputNodeName), 0};
    if(model->input->oper == NULL){
        return newTFInfo(model->status, TF_ABORTED, "could not find input node");
    }
    // assumes 1 output node
    model->output = malloc(sizeof(TF_Output));
    *model->output = (TF_Output){TF_GraphOperationByName(model->graph, modelInfo->outputNodeName), 0};
    if(model->output->oper == NULL)
    {
       return newTFInfo(model->status, TF_ABORTED, "could not find output node"); 
    }
    return (TFInfo){code:TF_OK, status:NULL};
}
void freeModel(Model* model){
    TF_DeleteGraph(model->graph);
    TF_DeleteSession(model->session, model->status);
    TF_DeleteSessionOptions(model->sessionOptions);
    TF_DeleteStatus(model->status);
    free(model->output);
    free(model->input);
    free(model);
}
TFInfo createTensor(TF_Tensor** inputTensor, TF_Status *status){
    inputTensor = (TF_Tensor**)malloc(sizeof(TF_Tensor*));
    int numOfDims=3;
    int64_t dims[]= {1,2,2};
    float data[1][2][2]= {{{1.0f,1.0f},{30.0f,80.0f}}};
    int dataSize = sizeof(data);
    TF_Tensor* tensor = TF_NewTensor(TF_FLOAT, dims, numOfDims, data, dataSize, &NoOpDeallocator, 0);
    if(tensor == NULL){
        return newTFInfo(status, TF_ABORTED, "Input tensor could not be created");
    }
    inputTensor[0] = tensor;
    return newTFInfo(status, TF_OK, "");
}
TFInfo run(Model* model, TF_Tensor** tensor){
     TF_Tensor** OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*));
TF_SessionRun(model->session, NULL, model->input, tensor, 1, model->output, OutputValues, 1, NULL, 0,NULL , model->status);
return(TFInfo){code:TF_GetCode(model->status), status: model->status};
}
int main(){
    Model* model = newModel();
    ModelInfo* noiseCancel = &supportedModels[NoiseCancel];
    TF_Code code = TF_OK;
    code=logStatus(loadModel(model, noiseCancel->directoryLocation, noiseCancel->tag), "loading model");
    if(code){
        return code;
    }
    code=logStatus(findModelNodes(model, noiseCancel), "finding graph nodes");
    if(code){
        return code;
    }
    TF_Tensor** tensor;
    code=logStatus(createTensor(tensor, model->status), "input tensor");
    if(code){
        return code;
    }
    code=logStatus(run(model, tensor), "running model");
    if(code){
        return code;
    }
    freeModel(model);
    return 0;
}