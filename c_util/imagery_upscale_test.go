package c_util

import (
	"image"
	"image/png"
	"os"
	"testing"
)

func loadImage(filename string) (*image.RGBA, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	imageData, err := png.Decode(file)
	if err != nil {
		return nil, err
	}

	rgbaImage, ok := imageData.(*image.RGBA)
	if !ok {
		return nil, err
	}
	return rgbaImage, nil
}

// TestInit tests the init function for the c code (which loads the models and introspects them)
func TestInit(t *testing.T) {
	err := LoadImageUpscaleLibrary()
	if err != nil {
		t.Fatal(err)
	}
}

// TestNoiseCancel runs noise cancel model on image
func TestNoiseCancel(t *testing.T) {
	img, err := loadImage("../test_image.png")
	if err != nil {
		t.Fatal(err)
	}
	scaledImg := UpscaleImage(img, C_NOISE_CANCEL)

	if scaledImg.Bounds().Max.X == img.Bounds().Max.X &&
		scaledImg.Bounds().Max.Y == img.Bounds().Max.Y {
		t.Fatalf("failed to scale image using Noise_Cancel model")
	}
}

// TestGAN run GAN model on image
func TestGAN(t *testing.T) {
	img, err := loadImage("../test_image.png")
	if err != nil {
		t.Fatal(err)
	}
	scaledImg := UpscaleImage(img, C_GAN)

	if scaledImg.Bounds().Max.X == img.Bounds().Max.X &&
		scaledImg.Bounds().Max.Y == img.Bounds().Max.Y {
		t.Fatalf("failed to scale image using GAN model")
	}
}
