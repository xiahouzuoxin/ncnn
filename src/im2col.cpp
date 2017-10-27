
#include <cstring>
#include "im2col.h"

namespace ncnn {

// Code From: https://github.com/intel/caffe/blob/master/src/caffe/util/im2col.cpp

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  int dil_kernel_h = (kernel_h - 1) * dilation_h + 1;
  int dil_kernel_w = (kernel_w - 1) * dilation_w + 1;
  int height_col = (height + 2 * pad_h - dil_kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - dil_kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;

    const int hc0 = h_offset * dilation_h - pad_h;
    const int wc0 = w_offset * dilation_w - pad_w;
    for (int h = 0; h < height_col; ++h) {
      int h_pad = h * stride_h + hc0;

      const int row_offset = (c * height_col + h) * width_col;
      const int srow_offset = (c_im * height + h_pad) * width;
      for (int w = 0; w < width_col; ++w) {
        int w_pad = w * stride_w + wc0;
        if ((((unsigned)h_pad) < ((unsigned)height)) && (((unsigned)w_pad) < ((unsigned)width)))
          data_col[row_offset + w] = data_im[srow_offset + w_pad];
        else {
          data_col[row_offset + w] = 0.;
        }
      }
    }
  }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_col);


template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  int dil_patch_h = (kernel_h - 1) * dilation_h + 1;
  int dil_patch_w = (kernel_w - 1) * dilation_w + 1;
  int height_col = (height + 2 * pad_h - dil_patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - dil_patch_w) / stride_w + 1;
  long chunk_len = kernel_h * kernel_w;

  memset(data_im, 0, height * width * channels * sizeof(Dtype));

  #ifdef _OPENMP
  #pragma omp parallel for if (channels > 1)
  #endif 
  for (int idx = 0; idx < channels; ++idx) {
    for (int inner_idx = 0; inner_idx < chunk_len; ++inner_idx) {
      int c = idx * chunk_len + inner_idx;
      int w_offset = c % kernel_w;
      int h_offset = (c / kernel_w) % kernel_h;
      int c_im = c / kernel_h / kernel_w;

      const int hc0 = h_offset * dilation_h - pad_h;
      const int wc0 = w_offset * dilation_w - pad_w;
      for (int h = 0; h < height_col; ++h) {
        for (int w = 0; w < width_col; ++w) {
          int h_pad = h * stride_h + hc0;
          const int srow_offset = (c_im * height + h_pad) * width;
          const int row_offset = (c * height_col + h) * width_col;
          int w_pad = w * stride_w + wc0;
          if ((((unsigned)h_pad) < ((unsigned)height)) && (((unsigned)w_pad) < ((unsigned)width))) {
            data_im[srow_offset + w_pad] += data_col[row_offset + w];
          }
        }
      }
    }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im);

} // namespace ncnn

