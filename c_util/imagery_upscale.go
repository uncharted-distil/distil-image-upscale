package c_util

/*
#cgo CFLAGS: -I ../include
#cgo linux LDFLAGS: -ltensorflow
#cgo darwin LDFLAGS: -ltensorflow
#cgo windows CFLAGS: -IC:/usr/local/include
#cgo windows LDFLAGS: -ltensorflow -LC:/usr/local/lib
#include "../src/entry_functions.c"
#include <stdio.h>
float *buffer;
// assumes 4 dimensions because it is required
void createBuffer(int64_t* dimensions){
	float (*arr)[dimensions[1]][dimensions[2]][dimensions[3]] = malloc(dimensions[0] * sizeof(*arr));
	buffer = &arr[0][0][0][0];
}
int rowMajorIndex(int x, int y, int z, int ySize, int zSize){
	return z + zSize * (y + ySize * x);
}
void setBuffer(unsigned int x,unsigned int y, float r, float g,float b, unsigned int ySize, unsigned int depthSize){
	float *p = buffer + rowMajorIndex(x,y,0,ySize,depthSize);
	*p=r;
	p = buffer + rowMajorIndex(x,y,1,ySize,depthSize);
	*p=g;
	p = buffer + rowMajorIndex(x,y,2,ySize,depthSize);
	*p=b;
}
void printBuffer(unsigned int x, unsigned int y, unsigned int depth){
	for(unsigned int i=0; i < x; ++i)
	{
		for(unsigned int ii=0; ii < y; ++ii)
		{
			printf("[%f,%f,%f]\n",*(buffer + rowMajorIndex(i,ii,0,y,depth)),*(buffer + rowMajorIndex(i,ii,1,y,depth)),*(buffer + rowMajorIndex(i,ii,2,y,depth)));
			fflush(stdout);
		}
	}
}
void* getBuffer(){
	return buffer;
}
void printOutput(float* data, unsigned int numOfDimensions, int64_t* dimensions){
	for(unsigned int i=0; i < numOfDimensions; ++i)
	{
		printf("%i \n", (int)dimensions[i]);
		fflush(stdout);
	}
	for(unsigned int x=0; x < dimensions[1]; ++x)
	{
		for(unsigned int y=0; y < dimensions[2]; ++y)
		{
			int r = rowMajorIndex(x,y,0, dimensions[2], dimensions[3]);
			int g = rowMajorIndex(x,y,1, dimensions[2], dimensions[3]);
			int b = rowMajorIndex(x,y,2, dimensions[2], dimensions[3]);
			printf("[%f,%f,%f]", data[r], data[g], data[b]);
			fflush(stdout);
		}
	}
}
// used for indexing float arrays
float getValueF(float* buffer, int index)
{
	return buffer[index];
}
// used for indexing int64_t
int getValueI64(int64_t* buffer, int index)
{
	return (int)buffer[index];
}
// free memory
void freeBuffer(){
	free(buffer);
}
// size of float
unsigned int floatSize(){
	return sizeof(float);
}
*/
import "C"
import (
	"fmt"
	"image"
	"image/draw"
	"math"
	"unsafe"

	"github.com/disintegration/imaging"
	"github.com/pkg/errors"
	log "github.com/unchartedsoftware/plog"
)

// ModelType defines the models to run for image-upscale
type ModelType int

const (
	C_NOISE_CANCEL ModelType = iota
	C_GAN
)
// GetModelType returns the current supported types and prevents wrong indexing
func GetModelType(val int) ModelType {
	switch val {
	case 0:
		return C_NOISE_CANCEL
	case 1:
		return C_GAN
	default:
		return C_GAN
	}
}

// LoadImageUpscaleLibrary loads the model for image upscaling
func LoadImageUpscaleLibrary(model ModelType) error {
	// buffer to hold error messages
	buffer := make([]byte, 256)
	errorMsg := (*C.char)(unsafe.Pointer(&buffer[0]))
	// loads the models into memory
	C.initialize(errorMsg, C.ModelTypes(model))

	goErrMsg := C.GoString(errorMsg)
	// if errorMsg is empty initialize was successful
	if goErrMsg == "" {
		log.Infof("image-upscale loaded.")
		return nil
	}
	err := errors.New(fmt.Sprintf("Failed to load image-upscale: %s", goErrMsg))
	log.Error(err)
	return err
}

// UpscaleImage upscales the supplied image through the use machine learning
func UpscaleImage(img *image.RGBA, modelType ModelType) *image.RGBA {
	// buffer to hold error messages
	buffer := make([]byte, 256)
	// cast to c char *
	errorMsg := (*C.char)(unsafe.Pointer(&buffer[0]))
	// build meta information needed for model
	colorDepth := 3
	imgSize := img.Bounds().Max
	// dimension of input {batchSize, width, height, colorDepth}
	dimBuffer := []int64{1, int64(imgSize.X), int64(imgSize.Y), int64(colorDepth)}
	//
	dimensions := (*C.int64_t)(unsafe.Pointer(&dimBuffer[0]))
	C.createBuffer(dimensions)
	// free the memory
	defer C.freeBuffer()
	// decode the *image.RGBA into raw format in the C buffer
	decodeToRaw(img)
	// building meta info again
	dataSize := C.uint(imgSize.X * imgSize.Y * colorDepth * int(C.floatSize()))
	dataInput := C.DataInfo{numberOfDimensions: C.uint(4), dimensions: dimensions, dataType: C.TF_FLOAT, dataSize: dataSize, data: C.getBuffer()}
	// run the model
	output := C.runModel(errorMsg, C.ModelTypes(modelType), dataInput)
	// if errorMsg is not empty there was an error
	if C.GoString(errorMsg) != "" {
		log.Error(errors.New(C.GoString(errorMsg)))
		return img
	}
	// get output dimension
	y := C.getValueI64(output.dimension, 2)
	x := C.getValueI64(output.dimension, 1)
	// encode raw into go *image.RGBA
	newImg := encodeToImage([2]int{int(y), int(x)}, output.buffer)
	C.freeOutputData(output)
	return newImg
}

// decodes *image.RGBA into raw then puts the raw into C memory
func decodeToRaw(img *image.RGBA) {
	colorDepth := 3
	imgSize := img.Bounds().Max
	maxSize := 255.0
	for x := 0; x < imgSize.X; x++ {
		for y := 0; y < imgSize.Y; y++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// normalize (0-255) -> (0.0f-1.0f)
			fR := float64(r>>8) / maxSize
			fG := float64(g>>8) / maxSize
			fB := float64(b>>8) / maxSize
			// set memory in C
			C.setBuffer(C.uint(x), C.uint(y), C.float(fR), C.float(fG), C.float(fB), C.uint(imgSize.Y), C.uint(colorDepth))
		}
	}
}

// the model can produce values outside of 0.0f-1.0f so clamp it to avoid rollover
func clamp(min float64, max float64, value float64) float64 {
	val := math.Min(max, value)
	return math.Max(min, val)
}

// reads from C memory and populates an *image.RGBA
func encodeToImage(dimension [2]int, buffer *C.float) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, dimension[0], dimension[1]))
	step := 4
	idx := 0
	min := 0.0
	max := 1.0
	maxValue := 255.0
	for x := 0; x < dimension[0]; x++ {
		for y := 0; y < dimension[1]; y++ {
			cX := C.int(x)
			cY := C.int(y)
			// gets index based on rowMajor contiguous memory
			rI := C.rowMajorIndex(cX, cY, 0, C.int(dimension[1]), 3)
			gI := C.rowMajorIndex(cX, cY, 1, C.int(dimension[1]), 3)
			bI := C.rowMajorIndex(cX, cY, 2, C.int(dimension[1]), 3)
			// gets values at indices
			r := C.getValueF(buffer, rI)
			g := C.getValueF(buffer, gI)
			b := C.getValueF(buffer, bI)
			// clamp 0.0f-1.0f and convert to 0-255 uint8
			img.Pix[idx] = uint8(clamp(min, max, float64(r)) * maxValue)
			img.Pix[idx+1] = uint8(clamp(min, max, float64(g)) * maxValue)
			img.Pix[idx+2] = uint8(clamp(min, max, float64(b)) * maxValue)
			img.Pix[idx+3] = uint8(maxValue)
			idx += step
		}
	}

	tImg := imaging.Transpose(img)
	draw.Draw(img, img.Bounds(), tImg, tImg.Bounds().Min, draw.Src)
	return img
}

