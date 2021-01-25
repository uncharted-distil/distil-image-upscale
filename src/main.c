#include "models.c"
#include "macros.c"
#include "utility.c"
#include <stdio.h>

// int main(){
//     Model* model = newModel();
//     ModelInfo* noiseCancel = &supportedModels[NoiseCancel];
//     TF_Code code = TF_OK;
//     code=logStatus(loadModel(model, noiseCancel->directoryLocation, noiseCancel->tag), "loading model");
//     if(code){
//         return code;
//     }
//     code=logStatus(findModelNodes(model, noiseCancel), "finding graph nodes");
//     if(code){
//         return code;
//     }
//     TF_Tensor** tensor=(TF_Tensor**)malloc(sizeof(TF_Tensor*));
//     TF_Tensor* outputTensor=malloc(sizeof(TF_Tensor*));
//    for(unsigned int i=0; i < noiseCancel->numStaticInputData; ++i){
//        TFInfo check = dataInfoToTensor(&tensor, &noiseCancel->staticInputData[i], model->status, i);
//        if(check.code){
//            freeTensor(tensor, noiseCancel->numInputNodes);
//            free(outputTensor);
//            return NULL;
//        }
//    }
//     float data[1][3][3][3]= {{{{1.0f,1.0f,1.0f},{30.0f,80.0f,60.0f},{20.0f,10.0f,120.0f}},
//     {{1.0f,1.0f,1.0f},{30.0f,80.0f,60.0f},{20.0f,10.0f,120.0f}},
//     {{1.0f,1.0f,1.0f},{30.0f,80.0f,60.0f},{20.0f,10.0f,120.0f}}}};
//     DataInfo data1;
//     data1.data = data;
//     data1.dataSize = 27 * sizeof(float);
//     data1.dataType = TF_FLOAT;
//     int64_t dim [] = {1,3,3,3}; 
//     data1.dimensions = dim;
//     data1.numberOfDimensions = 4;
//     code=logStatus(dataInfoToTensor(&tensor, &data1, model->status, 1), "input tensor");
//     if(code){
//         return code;
//     }
//     code=logStatus(run(model, noiseCancel, tensor, &outputTensor), "running model");
//     if(code){
//         return code;
//     }
//     float* buffer = (float *)TF_TensorData(outputTensor);
//     int numDims = TF_NumDims(outputTensor);
//     printf("%i \n", numDims);
//     for(unsigned int i=0; i < numDims; ++i){
//         printf("%i \n", TF_Dim(outputTensor, i));
//     }
//     freeModel(model);
//     return 0;
// }
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

EXPORTED float* runModel(const char* errorMsg, DataInfo dataInfo){
    TF_Tensor** inputTensor=(TF_Tensor**)malloc(sizeof(TF_Tensor*) * modelInfo->numInputNodes);
    TF_Tensor* outputTensor = malloc(sizeof(TF_Tensor*));
    for(unsigned int i=0; i < modelInfo->numStaticInputData; ++i){
        TFInfo check = dataInfoToTensor(&inputTensor, &modelInfo->staticInputData[i], model->status, i);
        if(check.code){
            errorMsg = TF_Message(check.status);
            freeTensor(inputTensor, modelInfo->numInputNodes);
            free(outputTensor);
            return NULL;
        }
    }
    TFInfo check = dataInfoToTensor(&inputTensor, &dataInfo, model->status, modelInfo->numStaticInputData);
    if(check.code){
        errorMsg = TF_Message(check.status);
        freeTensor(inputTensor, modelInfo->numInputNodes);
        free(outputTensor);
        return NULL;
    }
    check = run(model, modelInfo, inputTensor, &outputTensor);
    if(check.code){
        errorMsg = TF_Message(check.status);
        freeTensor(inputTensor, modelInfo->numInputNodes);
        freeTensor(&outputTensor, 1);
        return NULL;
    }
    float* buffer = (float *)TF_TensorData(outputTensor);
    
    return buffer;
}

EXPORTED void cleanup(){
    freeModel(model);
}