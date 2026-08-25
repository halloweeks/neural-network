## Neural Network

This project is a simple fully connected neural network written from scratch in C.

The network is trained to learn facial image patterns and generate new **28×28 RGB face images**. It is a small experimental model created for learning and exploring the fundamentals of neural networks and image generation.

It is **not intended to produce hyper-realistic images**. The generated faces are simple and low-resolution, but the model demonstrates how a neural network can learn visual patterns from training data and generate new outputs.

## Model Details

- **Architecture:** Fully Connected Autoencoder (Encoder + Decoder)
- **Network:** `2352 → 256 → 32 → 256 → 2352`
- **Input:** 28×28 RGB image
- **Output:** 28×28 RGB image
- **Hidden Layer 1:** 256 neurons
- **Latent Layer:** 32 neurons
- **Hidden Layer 2:** 256 neurons
- **Total Parameters:** 1,223,504
- **Total Weights:** 1,220,608
- **Total Biases:** 2,896
- **Encoder Parameters:** 610,592
- **Decoder Parameters:** 612,912
- **Model Size:** ~4.67 MiB
- **Data Type:** 32-bit floating point (`float`)
- **Activation:** Sigmoid
- **Model File:** `model.bin`
- **Implementation:** C
- **Framework:** None

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
