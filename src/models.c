#pragma once
#include "type_defines.c"
// enum for supportedModels index
typedef enum ModelTypes{
    // NoiseCancel model is an RDN that doubles resolution and removes artifacts
    NoiseCancel,
    // will support GAN at some point (there are some issues with the model export)
    GAN,
} ModelTypes;
const char *noiseCancelNodeNames[2]={
            "saver_filename", // input node that requires string pointing to the variables file of the saved model
            "serving_default_LR", // input node to receive the float image data
        };
const char *ganNodeNames[2]={
    "saver_filename",
    "serving_default_LR_input",
};
int64_t noiseCancelDims[1]={1}; 
DataInfo noiseCancelDataInfo[1] = {{
    .data="..models/noise_cancel/variables/variables",
    .dataSize= 60,
    .dataType= TF_STRING,
    .numberOfDimensions= 1,
    .dimensions= noiseCancelDims,
}};
DataInfo ganDataInfo[1] = {{
    .data="../models/gan_model/variables/variables",
    .dataSize= 60,
    .dataType= TF_STRING,
    .numberOfDimensions= 1,
    .dimensions= noiseCancelDims,
}};
// All supported models add to the below object
ModelInfo supportedModels[2] = {
    {
        .directoryLocation="../models/noise_cancel", 
        .inputNodeNames= noiseCancelNodeNames,
        .outputNodeName="StatefulPartitionedCall", 
        .tag="serve",
        .numInputNodes=2,
        .staticInputData= noiseCancelDataInfo,
        .numStaticInputData=1, // should always be numInputNodes - 1 for noise_cancel
    }, 
    {
        .directoryLocation="../models/gan_model", 
        .inputNodeNames= ganNodeNames,
        .outputNodeName="StatefulPartitionedCall", 
        .tag="serve",
        .numInputNodes=2,
        .staticInputData= ganDataInfo,
        .numStaticInputData=1, // should always be numInputNodes - 1 for noise_cancel 
    }
};
