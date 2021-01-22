#include <tensorflow/c/c_api.h>
// TF_Info used to hold status and error codes from tensorflow
typedef struct TF_Info{
    // code denoting error state
    TF_Code code;
    // status is useful for logging the error message
    TF_Status *status;
}TFInfo;

// ModelInfo is used to hold the necessary information for a model
// Note: use saved_model_cli tensorflow tool to gain insight into the input and output node names
// https://www.tensorflow.org/guide/saved_model#install_the_savedmodel_cli
typedef struct ModelInfo{
    // folder location to load
    const char* directoryLocation;
    // inputNodeName is used to define the entry point of the model for the data
    const char* inputNodeName;
    // outputNodeName is used to define the exit point of the model (where to receive the data)
    const char* outputNodeName;
    // tag is required to find the nodeNames
    const char* tag;
}ModelInfo;

typedef struct Model{
    // session holds 
    TF_Session* session;
    TF_Graph* graph;
    TF_Status* status;
    TF_SessionOptions* sessionOptions;
    TF_Buffer* runOptions;
    TF_Output* input;
    TF_Output* output;
}Model;