#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "pngwriter.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

/* Datatype for RGB pixel */
typedef struct {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
} pixel_t;

void cmap(double value, const double scaling, const double maxval,
          pixel_t * pix);

/* Heat colormap from black to white */
// *INDENT-OFF*
static int heat_colormap_old[256][3] = {
    {  0,   0,   0}, { 35,   0,   0}, { 52,   0,   0}, { 60,   0,   0},
    { 63,   1,   0}, { 64,   2,   0}, { 68,   5,   0}, { 69,   6,   0},
    { 72,   8,   0}, { 74,  10,   0}, { 77,  12,   0}, { 78,  14,   0},
    { 81,  16,   0}, { 83,  17,   0}, { 85,  19,   0}, { 86,  20,   0},
    { 89,  22,   0}, { 91,  24,   0}, { 92,  25,   0}, { 94,  26,   0},
    { 95,  28,   0}, { 98,  30,   0}, {100,  31,   0}, {102,  33,   0},
    {103,  34,   0}, {105,  35,   0}, {106,  36,   0}, {108,  38,   0},
    {109,  39,   0}, {111,  40,   0}, {112,  42,   0}, {114,  43,   0},
    {115,  44,   0}, {117,  45,   0}, {119,  47,   0}, {119,  47,   0},
    {120,  48,   0}, {122,  49,   0}, {123,  51,   0}, {125,  52,   0},
    {125,  52,   0}, {126,  53,   0}, {128,  54,   0}, {129,  56,   0},
    {129,  56,   0}, {131,  57,   0}, {132,  58,   0}, {134,  59,   0},
    {134,  59,   0}, {136,  61,   0}, {137,  62,   0}, {137,  62,   0},
    {139,  63,   0}, {139,  63,   0}, {140,  65,   0}, {142,  66,   0},  
    {148,  71,   0}, {149,  72,   0}, {149,  72,   0}, {151,  73,   0},
    {151,  73,   0}, {153,  75,   0}, {153,  75,   0}, {154,  76,   0},
    {154,  76,   0}, {154,  76,   0}, {156,  77,   0}, {156,  77,   0},
    {157,  79,   0}, {157,  79,   0}, {159,  80,   0}, {159,  80,   0},
    {159,  80,   0}, {160,  81,   0}, {160,  81,   0}, {162,  82,   0},
    {162,  82,   0}, {163,  84,   0}, {163,  84,   0}, {165,  85,   0},
    {165,  85,   0}, {166,  86,   0}, {166,  86,   0}, {166,  86,   0},
    {168,  87,   0}, {168,  87,   0}, {170,  89,   0}, {170,  89,   0},
    {171,  90,   0}, {171,  90,   0}, {173,  91,   0}, {173,  91,   0},
    {174,  93,   0}, {174,  93,   0}, {176,  94,   0}, {176,  94,   0},
    {177,  95,   0}, {177,  95,   0}, {179,  96,   0}, {179,  96,   0},
    {180,  98,   0}, {182,  99,   0}, {182,  99,   0}, {183, 100,   0},
    {183, 100,   0}, {185, 102,   0}, {185, 102,   0}, {187, 103,   0},
    {187, 103,   0}, {188, 104,   0}, {188, 104,   0}, {190, 105,   0},
    {191, 107,   0}, {191, 107,   0}, {193, 108,   0}, {193, 108,   0},
    {194, 109,   0}, {196, 110,   0}, {196, 110,   0}, {197, 112,   0},
    {197, 112,   0}, {199, 113,   0}, {200, 114,   0}, {200, 114,   0},
    {202, 116,   0}, {202, 116,   0}, {204, 117,   0}, {205, 118,   0},
    {205, 118,   0}, {207, 119,   0}, {208, 121,   0}, {208, 121,   0},
    {210, 122,   0}, {211, 123,   0}, {211, 123,   0}, {213, 124,   0},
    {214, 126,   0}, {214, 126,   0}, {216, 127,   0}, {217, 128,   0},
    {217, 128,   0}, {219, 130,   0}, {221, 131,   0}, {221, 131,   0},
    {222, 132,   0}, {224, 133,   0}, {224, 133,   0}, {225, 135,   0},
    {227, 136,   0}, {227, 136,   0}, {228, 137,   0}, {230, 138,   0},
    {230, 138,   0}, {231, 140,   0}, {233, 141,   0}, {233, 141,   0},
    {234, 142,   0}, {236, 144,   0}, {236, 144,   0}, {238, 145,   0},
    {239, 146,   0}, {241, 147,   0}, {241, 147,   0}, {242, 149,   0},
    {244, 150,   0}, {244, 150,   0}, {245, 151,   0}, {247, 153,   0},
    {247, 153,   0}, {248, 154,   0}, {250, 155,   0}, {251, 156,   0},
    {251, 156,   0}, {253, 158,   0}, {255, 159,   0}, {255, 159,   0},
    {255, 160,   0}, {255, 161,   0}, {255, 163,   0}, {255, 163,   0},
    {255, 164,   0}, {255, 165,   0}, {255, 167,   0}, {255, 167,   0},
    {255, 168,   0}, {255, 169,   0}, {255, 169,   0}, {255, 170,   0},
    {255, 172,   0}, {255, 173,   0}, {255, 173,   0}, {255, 174,   0},
    {255, 175,   0}, {255, 177,   0}, {255, 178,   0}, {255, 179,   0},
    {255, 181,   0}, {255, 181,   0}, {255, 182,   0}, {255, 183,   0},
    {255, 184,   0}, {255, 187,   7}, {255, 188,  10}, {255, 189,  14},
    {255, 191,  18}, {255, 192,  21}, {255, 193,  25}, {255, 195,  29},
    {255, 197,  36}, {255, 198,  40}, {255, 200,  43}, {255, 202,  51},
    {255, 204,  54}, {255, 206,  61}, {255, 207,  65}, {255, 210,  72},
    {255, 211,  76}, {255, 214,  83}, {255, 216,  91}, {255, 219,  98},
    {255, 221, 105}, {255, 223, 109}, {255, 225, 116}, {255, 228, 123},
    {255, 232, 134}, {255, 234, 142}, {255, 237, 149}, {255, 239, 156},
    {255, 240, 160}, {255, 243, 167}, {255, 246, 174}, {255, 248, 182},
    {255, 249, 185}, {255, 252, 193}, {255, 253, 196}, {255, 255, 204},
    {255, 255, 207}, {255, 255, 211}, {255, 255, 218}, {255, 255, 222},
    {255, 255, 225}, {255, 255, 229}, {255, 255, 233}, {255, 255, 236},
    {255, 255, 240}, {255, 255, 244}, {255, 255, 247}, {255, 255, 255}
};

static int heat_colormap[256][3] = {
    { 59,  76,  192 }, { 59,  76,  192 }, { 60,  78,  194 }, { 61,  80,  195 },
    { 62,  81,  197 }, { 64,  83,  198 }, { 65,  85,  200 }, { 66,  87,  201 },
    { 67,  88,  203 }, { 68,  90,  204 }, { 69,  92,  206 }, { 71,  93,  207 },
    { 72,  95,  209 }, { 73,  97,  210 }, { 74,  99,  211 }, { 75,  100, 213 },
    { 77,  102, 214 }, { 78,  104, 215 }, { 79,  105, 217 }, { 80,  107, 218 },
    { 82,  109, 219 }, { 83,  110, 221 }, { 84,  112, 222 }, { 85,  114, 223 },
    { 87,  115, 224 }, { 88,  117, 225 }, { 89,  119, 227 }, { 90,  120, 228 },
    { 92,  122, 229 }, { 93,  124, 230 }, { 94,  125, 231 }, { 96,  127, 232 },
    { 97,  129, 233 }, { 98,  130, 234 }, { 100, 132, 235 }, { 101, 133, 236 },
    { 102, 135, 237 }, { 103, 137, 238 }, { 105, 138, 239 }, { 106, 140, 240 },
    { 107, 141, 240 }, { 109, 143, 241 }, { 110, 144, 242 }, { 111, 146, 243 },
    { 113, 147, 244 }, { 114, 149, 244 }, { 116, 150, 245 }, { 117, 152, 246 },
    { 118, 153, 246 }, { 120, 155, 247 }, { 121, 156, 248 }, { 122, 157, 248 },
    { 124, 159, 249 }, { 125, 160, 249 }, { 127, 162, 250 }, { 128, 163, 250 },
    { 129, 164, 251 }, { 131, 166, 251 }, { 132, 167, 252 }, { 133, 168, 252 },
    { 135, 170, 252 }, { 136, 171, 253 }, { 138, 172, 253 }, { 139, 174, 253 },
    { 140, 175, 254 }, { 142, 176, 254 }, { 143, 177, 254 }, { 145, 179, 254 },
    { 146, 180, 254 }, { 147, 181, 255 }, { 149, 182, 255 }, { 150, 183, 255 },
    { 152, 185, 255 }, { 153, 186, 255 }, { 154, 187, 255 }, { 156, 188, 255 },
    { 157, 189, 255 }, { 158, 190, 255 }, { 160, 191, 255 }, { 161, 192, 255 },
    { 163, 193, 255 }, { 164, 194, 254 }, { 165, 195, 254 }, { 167, 196, 254 },
    { 168, 197, 254 }, { 169, 198, 254 }, { 171, 199, 253 }, { 172, 200, 253 },
    { 173, 201, 253 }, { 175, 202, 252 }, { 176, 203, 252 }, { 177, 203, 252 },
    { 179, 204, 251 }, { 180, 205, 251 }, { 181, 206, 250 }, { 183, 207, 250 },
    { 184, 207, 249 }, { 185, 208, 249 }, { 186, 209, 248 }, { 188, 209, 247 },
    { 189, 210, 247 }, { 190, 211, 246 }, { 191, 211, 246 }, { 193, 212, 245 },
    { 194, 213, 244 }, { 195, 213, 243 }, { 196, 214, 243 }, { 198, 214, 242 },
    { 199, 215, 241 }, { 200, 215, 240 }, { 201, 216, 239 }, { 202, 216, 239 },
    { 204, 217, 238 }, { 205, 217, 237 }, { 206, 217, 236 }, { 207, 218, 235 },
    { 208, 218, 234 }, { 209, 218, 233 }, { 210, 219, 232 }, { 211, 219, 231 },
    { 212, 219, 230 }, { 214, 220, 229 }, { 215, 220, 228 }, { 216, 220, 227 },
    { 217, 220, 225 }, { 218, 220, 224 }, { 219, 220, 223 }, { 220, 221, 222 },
    { 221, 221, 221 }, { 222, 220, 219 }, { 223, 220, 218 }, { 224, 219, 216 },
    { 225, 219, 215 }, { 226, 218, 214 }, { 227, 218, 212 }, { 228, 217, 211 },
    { 229, 216, 209 }, { 230, 216, 208 }, { 231, 215, 206 }, { 232, 215, 205 },
    { 233, 214, 203 }, { 233, 213, 202 }, { 234, 212, 200 }, { 235, 212, 199 },
    { 236, 211, 197 }, { 237, 210, 196 }, { 237, 209, 194 }, { 238, 208, 193 },
    { 239, 208, 191 }, { 239, 207, 190 }, { 240, 206, 188 }, { 240, 205, 187 },
    { 241, 204, 185 }, { 242, 203, 183 }, { 242, 202, 182 }, { 243, 201, 180 },
    { 243, 200, 179 }, { 243, 199, 177 }, { 244, 198, 176 }, { 244, 197, 174 },
    { 245, 196, 173 }, { 245, 195, 171 }, { 245, 194, 169 }, { 246, 193, 168 },
    { 246, 192, 166 }, { 246, 190, 165 }, { 246, 189, 163 }, { 247, 188, 161 },
    { 247, 187, 160 }, { 247, 186, 158 }, { 247, 184, 157 }, { 247, 183, 155 },
    { 247, 182, 153 }, { 247, 181, 152 }, { 247, 179, 150 }, { 247, 178, 149 },
    { 247, 177, 147 }, { 247, 175, 146 }, { 247, 174, 144 }, { 247, 172, 142 },
    { 247, 171, 141 }, { 247, 170, 139 }, { 247, 168, 138 }, { 247, 167, 136 },
    { 247, 165, 135 }, { 246, 164, 133 }, { 246, 162, 131 }, { 246, 161, 130 },
    { 246, 159, 128 }, { 245, 158, 127 }, { 245, 156, 125 }, { 245, 155, 124 },
    { 244, 153, 122 }, { 244, 151, 121 }, { 243, 150, 119 }, { 243, 148, 117 },
    { 242, 147, 116 }, { 242, 145, 114 }, { 241, 143, 113 }, { 241, 142, 111 },
    { 240, 140, 110 }, { 240, 138, 108 }, { 239, 136, 107 }, { 239, 135, 105 },
    { 238, 133, 104 }, { 237, 131, 102 }, { 237, 129, 101 }, { 236, 128, 99  },
    { 235, 126, 98  }, { 235, 124, 96  }, { 234, 122, 95  }, { 233, 120, 94  },
    { 232, 118, 92  }, { 231, 117, 91  }, { 230, 115, 89  }, { 230, 113, 88  },
    { 229, 111, 86  }, { 228, 109, 85  }, { 227, 107, 84  }, { 226, 105, 82  },
    { 225, 103, 81  }, { 224, 101, 79  }, { 223, 99,  78  }, { 222, 97,  77  },
    { 221, 95,  75  }, { 220, 93,  74  }, { 219, 91,  73  }, { 218, 89,  71  },
    { 217, 87,  70  }, { 215, 85,  69  }, { 214, 82,  67  }, { 213, 80,  66  },
    { 212, 78,  65  }, { 211, 76,  64  }, { 210, 74,  62  }, { 208, 71,  61  },
    { 207, 69,  60  }, { 206, 67,  59  }, { 204, 64,  57  }, { 203, 62,  56  },
    { 202, 59,  55  }, { 200, 57,  54  }, { 199, 54,  53  }, { 198, 52,  51  },
    { 196, 49,  50  }, { 195, 46,  49  }, { 193, 43,  48  }, { 192, 40,  47  },
    { 191, 37,  46  }, { 189, 34,  44  }, { 188, 30,  43  }, { 186, 26,  42  },
    { 185, 22,  41  }, { 183, 17,  40  }, { 182, 11,  39  }, { 180, 4,   38  },
};// *INDENT-ON*
/* Save the two dimensional array as a png image
 * Arguments:
 *   double *data is a pointer to an array of nx*ny values
 *   int nx is the number of COLUMNS to be written
 *   int ny is the number of ROWS to be written
 *   char *fname is the name of the picture
 *   char lang is either 'c' or 'f' denoting the memory
 *             layout. That is, if 'f' is given, then rows
 *             and columns are swapped.
 */
int save_png(float *data, const int height, const int width, const char *fname,
             const char lang)
{
    FILE *fp;
    png_structp pngstruct_ptr = NULL;
    png_infop pnginfo_ptr = NULL;
    png_byte **row_pointers = NULL;
    int i, j;

    /* Default return status is failure */
    int status = -1;

    int pixel_size = 3;
    int depth = 8;

    fp = fopen(fname, "wb");
    if (fp == NULL)
        goto fopen_failed;

    pngstruct_ptr =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (pngstruct_ptr == NULL)
        goto pngstruct_create_failed;

    pnginfo_ptr = png_create_info_struct(pngstruct_ptr);

    if (pnginfo_ptr == NULL)
        goto pnginfo_create_failed;

    if (setjmp(png_jmpbuf(pngstruct_ptr)))
        goto setjmp_failed;

    png_set_IHDR(pngstruct_ptr, pnginfo_ptr, (size_t) width,
                 (size_t) height, depth, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    row_pointers = (png_byte**)png_malloc(pngstruct_ptr, height * sizeof(png_byte *));

    for (i = 0; i < height; i++) {
        png_byte *row = (png_byte*)png_malloc(pngstruct_ptr,
                                   sizeof(uint8_t) * width * pixel_size);
        row_pointers[i] = row;

        // Branch according to the memory layout
        if (lang == 'c' || lang == 'C') {
            for (j = 0; j < width; j++) {
                pixel_t pixel;
                // Scale the values so that values between
                // 0 and 100 degrees are mapped to values
                // between 0 and 255
                cmap(data[j + i * width], 2.55, 0.0, &pixel);
                *row++ = pixel.red;
                *row++ = pixel.green;
                *row++ = pixel.blue;
            }
        } else {
            for (j = 0; j < width; j++) {
                pixel_t pixel;
                // Scale the values so that values between
                // 0 and 100 degrees are mapped to values
                // between 0 and 255
                cmap(data[i + j * height], 2.55, 0.0, &pixel);
                *row++ = pixel.red;
                *row++ = pixel.green;
                *row++ = pixel.blue;
            }
        }
    }

    png_init_io(pngstruct_ptr, fp);
    png_set_rows(pngstruct_ptr, pnginfo_ptr, row_pointers);
    png_write_png(pngstruct_ptr, pnginfo_ptr,
                  PNG_TRANSFORM_IDENTITY, NULL);

    status = 0;

    for (i = 0; i < height; i++) {
        png_free(pngstruct_ptr, row_pointers[i]);
    }
    png_free(pngstruct_ptr, row_pointers);

  setjmp_failed:
  pnginfo_create_failed:
    png_destroy_write_struct(&pngstruct_ptr, &pnginfo_ptr);
  pngstruct_create_failed:
    fclose(fp);
  fopen_failed:
    return status;
}

/* This routine sets the RGB values for the pixel_t structure using
 * the colormap data heat_colormap. If the value is outside the
 * acceptable png values 0,255 blue or red color is used instead. */
void cmap(double value, const double scaling, const double offset,
          pixel_t * pix)
{
    int ival;

    ival = (int) (value * scaling + offset);
    if (ival < 0) {             // Colder than colorscale, substitute blue
        pix->red = 0;
        pix->green = 0;
        pix->blue = 255;
    } else if (ival > 255) {
        pix->red = 255;         // Hotter than colormap, substitute red
        pix->green = 0;
        pix->blue = 0;
    } else {
        pix->red = heat_colormap[ival][0];
        pix->green = heat_colormap[ival][1];
        pix->blue = heat_colormap[ival][2];
    }
}

// Function to load image file and return table
std::vector<std::vector<bool>> loadImage(const char* filename) {
    int width;
    int height;
    std::vector<std::vector<bool>> image;

    // Open the PNG file
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return {};
    }

    // Initialize PNG structures
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::cerr << "Error: png_create_read_struct failed" << std::endl;
        fclose(fp);
        return {};
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        std::cerr << "Error: png_create_info_struct failed" << std::endl;
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        return {};
    }

    // Set error handling
    if (setjmp(png_jmpbuf(png))) {
        std::cerr << "Error: Error during PNG read" << std::endl;
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        return {};
    }

    // Initialize PNG IO
    png_init_io(png, fp);
    png_read_info(png, info);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    int bit_depth = png_get_bit_depth(png, info);
    int color_type = png_get_color_type(png, info);

    // Ensure PNG image is in RGBA format
    if (color_type != PNG_COLOR_TYPE_RGBA) {
        std::cerr << "Error: PNG image is not in RGBA format" << std::endl;
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        return {};
    }

    // Allocate memory for image data
    png_bytep row_buffer = (png_bytep)malloc(png_get_rowbytes(png, info));
    if (!row_buffer) {
        std::cerr << "Error: Memory allocation failed" << std::endl;
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        return {};
    }

    // Read image data row by row
    image.resize(height, std::vector<bool>(width, false));
    for (int y = 0; y < height; ++y) {
        png_read_row(png, row_buffer, nullptr);
        png_bytep row = row_buffer;
        for (int x = 0; x < width; ++x) {
            // Check if the color is white (255, 255, 255, 255)
            if (row[1] == 0) {
                image[y][x] = true;
            } else {
                image[y][x] = false;
            }
            row += 4; // Move to the next pixel
        }
    }

    // Cleanup
    free(row_buffer);
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);

    return image;
}


void resizeImage(std::vector<std::vector<bool>>& table, int desiredWidth) {
    int originalWidth = table.empty() ? 0 : table[0].size();
    int originalHeight = table.size();

    // Calculate aspect ratio of original image
    double aspectRatio = static_cast<double>(originalWidth) / originalHeight;

    // Calculate the desired height based on the aspect ratio and desired width
    int desiredHeight = static_cast<int>(std::round(desiredWidth / aspectRatio));

    // Resize the table
    std::vector<std::vector<bool>> resizedTable(desiredHeight, std::vector<bool>(desiredWidth, false));
    for (int y = 0; y < desiredHeight; ++y) {
        for (int x = 0; x < desiredWidth; ++x) {
            // Calculate the corresponding position in the original table
            int originalX = static_cast<int>(x * static_cast<double>(originalWidth) / desiredWidth);
            int originalY = static_cast<int>(y * static_cast<double>(originalHeight) / desiredHeight);

            // Copy the value from the original table to the resized table
            resizedTable[y][x] = table[originalY][originalX];
        }
    }

    // Update the original table with the resized table
    table = std::move(resizedTable);
}
