#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#define WIDTH 28
#define HEIGHT 28
#define CHANNELS 3

#define INPUTS  (WIDTH * HEIGHT * CHANNELS)   // 2352
#define OUTPUTS (WIDTH * HEIGHT * CHANNELS)   // 2352
#define HIDDEN1 256
#define LATENT  32
#define HIDDEN2 256

float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

void generate_random_latent(float latent[LATENT])
{
    for (int i = 0; i < LATENT; i++) {
        latent[i] = (float)rand() / (float)RAND_MAX;
    }
}

void decode_latent(
	float latent[LATENT],
	float weights3[HIDDEN2][LATENT],
	float bias3[HIDDEN2],
	float weights4[OUTPUTS][HIDDEN2],
	float bias4[OUTPUTS],
	float output[OUTPUTS]) {
	
	float hidden2[HIDDEN2];
	
	// LATENT → HIDDEN2
	for (int j = 0; j < HIDDEN2; j++) {
		float sum = bias3[j];
		
		for (int i = 0; i < LATENT; i++) {
			sum += latent[i] * weights3[j][i];
		}
		
		hidden2[j] = sigmoid(sum);
	}
	
	// HIDDEN2 → OUTPUT
	for (int j = 0; j < OUTPUTS; j++) {
		float sum = bias4[j];
		
		for (int i = 0; i < HIDDEN2; i++) {
			sum += hidden2[i] * weights4[j][i];
		}
		
		output[j] = sigmoid(sum);
	}
}


void encode_input(
    const float input[INPUTS],
    const float weights1[HIDDEN1][INPUTS],
    const float bias1[HIDDEN1],
    const float weights2[LATENT][HIDDEN1],
    const float bias2[LATENT],
    float latent[LATENT]
)
{
    float hidden1[HIDDEN1];

    // Layer 1
    for (int i = 0; i < HIDDEN1; i++) {
        float sum = bias1[i];

        for (int j = 0; j < INPUTS; j++) {
            sum += weights1[i][j] * input[j];
        }

        hidden1[i] = sigmoid(sum);
    }

    // Layer 2
    for (int i = 0; i < LATENT; i++) {
        float sum = bias2[i];

        for (int j = 0; j < HIDDEN1; j++) {
            sum += weights2[i][j] * hidden1[j];
        }

        latent[i] = sigmoid(sum);
    }
}

// ============================================================
// LOAD MODEL
// ============================================================

bool load_model(
	const char *filename,
	float weights1[HIDDEN1][INPUTS],
	float bias1[HIDDEN1],
	float weights2[LATENT][HIDDEN1],
	float bias2[LATENT],
	float weights3[HIDDEN2][LATENT],
	float bias3[HIDDEN2],
	float weights4[OUTPUTS][HIDDEN2],
	float bias4[OUTPUTS]) {
	
	FILE *file = fopen(filename, "rb");
	
	if (!file) {
		return false;
	}
	
	if (fread(weights1, sizeof(float), HIDDEN1 * INPUTS, file) != HIDDEN1 * INPUTS) {
		fclose(file);
		return false;
	}
	
	if (fread(bias1, sizeof(float), HIDDEN1, file) != HIDDEN1) {
		fclose(file);
		return false;
	}
	
	if (fread(weights2, sizeof(float), LATENT * HIDDEN1, file) != LATENT * HIDDEN1) {
		fclose(file);
		return false;
	}
	
	if (fread(bias2, sizeof(float), LATENT, file) != LATENT) {
		fclose(file);
		return false;
	}
	
	if (fread(weights3, sizeof(float), HIDDEN2 * LATENT, file) != HIDDEN2 * LATENT) {
		fclose(file);
		return false;
	}
	
	if (fread(bias3, sizeof(float), HIDDEN2, file) != HIDDEN2) {
		fclose(file);
		return false;
	}
	
	if (fread(weights4, sizeof(float), OUTPUTS * HIDDEN2, file) != OUTPUTS * HIDDEN2) {
		fclose(file);
		return false;
	}
	
	if (fread(bias4, sizeof(float), OUTPUTS, file) != OUTPUTS) {
		fclose(file);
		return false;
	}
	
	fclose(file);
	
	return true;
}

// Bitmap format used for simplicity
void save_bitmap2(const char *filename, float *output, int width, int height)
{
    FILE *f = fopen(filename, "wb");
    if (!f) return;

    int row_size = (width * 3 + 3) & ~3;
    int image_size = row_size * height;
    int file_size = 54 + image_size;

    unsigned char header[54] = {0};

    header[0] = 'B';
    header[1] = 'M';

    *(uint32_t *)&header[2]  = file_size;
    *(uint32_t *)&header[10] = 54;
    *(uint32_t *)&header[14] = 40;
    *(int32_t  *)&header[18] = width;
    *(int32_t  *)&header[22] = height;
    *(uint16_t *)&header[26] = 1;
    *(uint16_t *)&header[28] = 24;
    *(uint32_t *)&header[34] = image_size;

    fwrite(header, 1, 54, f);

    unsigned char *row = calloc(1, row_size);

    for (int y = height - 1; y >= 0; y--) {

        for (int x = 0; x < width; x++) {

            int i = (y * width + x) * 3;

            float r = output[i];
            float g = output[i + 1];
            float b = output[i + 2];

            if (r < 0.0f) r = 0.0f;
            if (r > 1.0f) r = 1.0f;

            if (g < 0.0f) g = 0.0f;
            if (g > 1.0f) g = 1.0f;

            if (b < 0.0f) b = 0.0f;
            if (b > 1.0f) b = 1.0f;

            // BMP stores BGR
            row[x * 3 + 0] = (unsigned char)(b * 255.0f);
            row[x * 3 + 1] = (unsigned char)(g * 255.0f);
            row[x * 3 + 2] = (unsigned char)(r * 255.0f);
        }

        fwrite(row, 1, row_size, f);
    }

    free(row);
    fclose(f);
}

int main(void)
{
	srand((unsigned int)time(NULL));
	
	// ========================================================
	// MODEL PARAMETERS
	// ========================================================
	float weights1[HIDDEN1][INPUTS];
	float weights2[LATENT][HIDDEN1];
	float weights3[HIDDEN2][LATENT];
	float weights4[OUTPUTS][HIDDEN2];
	
	float bias1[HIDDEN1];
	float bias2[LATENT];
	float bias3[HIDDEN2];
	float bias4[OUTPUTS];
	
	// ========================================================
	// LOAD
	// ========================================================
	bool mload = load_model(
		"model.bin",
		weights1,
		bias1,
		weights2,
		bias2,
		weights3,
		bias3,
		weights4,
		bias4
	);
	
	if (!mload) {
		printf("Failed to load model\n");
		return 1;
	}
	
	printf("Model loaded successfully!\n");
	
	
	// ========================================================
	// RUN
	// ========================================================
	float output[OUTPUTS];
	float latent[LATENT];
	
	char filename[256];
	
	if (mkdir("output", 0755) == -1) {
		if (errno != EEXIST) {
			perror("Failed to create output directory");
			return 1;
		}
		
		struct stat st;
		
		if (stat("output", &st) == -1) {
			perror("Failed to stat output");
			return 1;
		}
		
		if (!S_ISDIR(st.st_mode)) {
			fprintf(stderr, "'output' exists but is not a directory\n");
			return 1;
		}
	}
	
	// Generate sample images 
	for (int sample = 0; sample < 10; sample++) {
		generate_random_latent(latent);
		
		decode_latent(
			latent,
			weights3,
			bias3,
			weights4,
			bias4,
			output
		);
		
		snprintf(filename, sizeof(filename), "output/output_%02d.bmp", sample + 1);
		
		// save output image file in .bmp
		save_bitmap2(filename, output, WIDTH, HEIGHT);
		
		printf("Saved: output/output_%02d.bmp\n", sample + 1);
		
	}
	
	return 0;
}