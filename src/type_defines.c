#pragma once
#include "../include/tensorflow/c/c_api.h"

// TF_Info used to hold status and error codes from tensorflow
typedef struct TF_Info{
    // code denoting error state
    TF_Code code;
    // status is useful for logging the error message
    TF_Status *status;
}TFInfo;

// Data is a struct used to contain all of the data information
typedef struct DataInfo{
    // number of dimensions in the data noise_cancel requires {-1,-1,-1,3} which is 4 dimensions
    unsigned int numberOfDimensions;
    // denotes size of each dimension
    int64_t* dimensions;
    // should be the number of elements in data
    unsigned int dataSize;
    // should be float for images
    TF_DataType dataType;
    // data pointer
    void* data;
}DataInfo;

// ModelInfo is used to hold the necessary information for a model
// Note: use saved_model_cli tensorflow tool to gain insight into the input and output node names
// https://www.tensorflow.org/guide/saved_model#install_the_savedmodel_cli
typedef struct ModelInfo{
    // folder location to load
    const char* directoryLocation;
    // inputNodeName is used to define the entry points of the model for the data
    const char** inputNodeNames;
    // outputNodeName is used to define the exit point of the model (where to receive the data)
    const char* outputNodeName;
    // tag is required to find the nodeNames
    const char* tag;
    // should point to variables path for the exported model
    const char* variables;
    // number of inputNodeNames
    unsigned int numInputNodes;
    // some models have static information that needs to be passed in every run call
    DataInfo* staticInputData;
    // number of staticInputData
    unsigned int numStaticInputData;
}ModelInfo;

// holds all the necessary pointers to run the model
typedef struct Model{
    // session holds 
    TF_Session* session;
    // graph is the model loaded from folder
    TF_Graph* graph;
    // status is used to contain msgs for errors
    TF_Status* status;
    // sessionOptions such as specific device
    TF_SessionOptions* sessionOptions;
    // not needed
    TF_Buffer* runOptions;
    // input node for the graph
    TF_Output* input;
    // output node for the graph
    TF_Output* output;
}Model;

// OutputData contains all the necessary information to reconstruct the image
typedef struct OutputData{
    // buffer to contain the data
    float* buffer;
    // number of dimensions
    unsigned int numOfDimensions;
    // shape of the data
    int64_t* dimension;
}OutputData;