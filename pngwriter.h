#ifndef PNGWRITER_H_
#define PNGWRITER_H_

#include <vector>
int save_png(float *data, const int nx, const int ny, const char *fname,
             const char lang);
std::vector<std::vector<int>> loadImage(const char* filename);

void resizeImage(std::vector<std::vector<int>>& table, int desiredWidth);
#endif
