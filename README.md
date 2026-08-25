## Neural Network

This project is a simple fully connected neural network written from scratch in C.

The network is trained to learn facial image patterns and generate new **28×28 RGB face images**. It is a small experimental model created for learning and exploring the fundamentals of neural networks and image generation.

It is **not intended to produce hyper-realistic images**. The generated faces are simple and low-resolution, but the model demonstrates how a neural network can learn visual patterns from training data and generate new outputs.

## Model Details

- **Architecture:** Fully Connected Neural Network
- **Input:** 28×28 RGB image
- **Output:** 28×28 RGB image
- **Parameters:** ~1.2 million
- **Model size:** ~4.66 MB
- **Model format:** Binary (`model.bin`)
- **Implementation:** C
- **Framework:** None
- **Image format:** BMP
- **Image resolution:** 28×28

### Build

Compile with GCC:

```bash
cc model.c -o model -lm
```

## Generated Samples

<p align="center">
  <img src="output/output_01.bmp" width="40">
  <img src="output/output_02.bmp" width="40">
  <img src="output/output_03.bmp" width="40">
  <img src="output/output_04.bmp" width="40">
  <img src="output/output_05.bmp" width="40">
</p>

<p align="center">
  <img src="output/output_06.bmp" width="40">
  <img src="output/output_07.bmp" width="40">
  <img src="output/output_08.bmp" width="40">
  <img src="output/output_09.bmp" width="40">
  <img src="output/output_10.bmp" width="40">
</p>
