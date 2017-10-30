// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "deconvolution.h"
#include "im2col.h"
#if NCNN_EIGEN
#include "Eigen/Dense"
#endif

namespace ncnn {

DEFINE_LAYER_CREATOR(Deconvolution)

Deconvolution::Deconvolution()
{
    one_blob_only = true;
    support_inplace = false;
}

Deconvolution::~Deconvolution()
{
}

int Deconvolution::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_size = pd.get(1, 0);
    dilation = pd.get(2, 1);
    stride = pd.get(3, 1);
    pad = pd.get(4, 0);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);

    return 0;
}

#if NCNN_STDIO
int Deconvolution::load_model(FILE* binfp)
{
    int nread;

    union
    {
        struct
        {
            unsigned char f0;
            unsigned char f1;
            unsigned char f2;
            unsigned char f3;
        };
        unsigned int tag;
    } flag_struct;

    nread = fread(&flag_struct, sizeof(flag_struct), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "Deconvolution read flag_struct failed %d\n", nread);
        return -1;
    }

    unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

    weight_data.create(weight_data_size);
    if (weight_data.empty())
        return -100;

    if (flag_struct.tag == 0x01306B47)
    {
        // half-precision weight data
        int align_weight_data_size = alignSize(weight_data_size * sizeof(unsigned short), 4);
        std::vector<unsigned short> float16_weights;
        float16_weights.resize(align_weight_data_size);
        nread = fread(float16_weights.data(), align_weight_data_size, 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Deconvolution read float16_weights failed %d\n", nread);
            return -1;
        }

        weight_data = Mat::from_float16(float16_weights.data(), weight_data_size);
        if (weight_data.empty())
            return -100;
    }
    else if (flag != 0)
    {
        // quantized weight data
        float quantization_value[256];
        nread = fread(quantization_value, 256 * sizeof(float), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Deconvolution read quantization_value failed %d\n", nread);
            return -1;
        }

        int align_weight_data_size = alignSize(weight_data_size * sizeof(unsigned char), 4);
        std::vector<unsigned char> index_array;
        index_array.resize(align_weight_data_size);
        nread = fread(index_array.data(), align_weight_data_size, 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Deconvolution read index_array failed %d\n", nread);
            return -1;
        }

        float* weight_data_ptr = weight_data;
        for (int i = 0; i < weight_data_size; i++)
        {
            weight_data_ptr[i] = quantization_value[ index_array[i] ];
        }
    }
    else if (flag_struct.f0 == 0)
    {
        // raw weight data
        nread = fread(weight_data, weight_data_size * sizeof(float), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Deconvolution read weight_data failed %d\n", nread);
            return -1;
        }
    }

    if (bias_term)
    {
        bias_data.create(num_output);
        if (bias_data.empty())
            return -100;
        nread = fread(bias_data, num_output * sizeof(float), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Deconvolution read bias_data failed %d\n", nread);
            return -1;
        }
    }

    return 0;
}
#endif // NCNN_STDIO

int Deconvolution::load_model(const unsigned char*& mem)
{
    union
    {
        struct
        {
            unsigned char f0;
            unsigned char f1;
            unsigned char f2;
            unsigned char f3;
        };
        unsigned int tag;
    } flag_struct;

    memcpy(&flag_struct, mem, sizeof(flag_struct));
    mem += sizeof(flag_struct);

    unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

    if (flag_struct.tag == 0x01306B47)
    {
        // half-precision weight data
        weight_data = Mat::from_float16((unsigned short*)mem, weight_data_size);
        mem += alignSize(weight_data_size * sizeof(unsigned short), 4);
        if (weight_data.empty())
            return -100;
    }
    else if (flag != 0)
    {
        // quantized weight data
        const float* quantization_value = (const float*)mem;
        mem += 256 * sizeof(float);

        const unsigned char* index_array = (const unsigned char*)mem;
        mem += alignSize(weight_data_size * sizeof(unsigned char), 4);

        weight_data.create(weight_data_size);
        if (weight_data.empty())
            return -100;
        float* weight_data_ptr = weight_data;
        for (int i = 0; i < weight_data_size; i++)
        {
            weight_data_ptr[i] = quantization_value[ index_array[i] ];
        }
    }
    else if (flag_struct.f0 == 0)
    {
        // raw weight data
        weight_data = Mat(weight_data_size, (float*)mem);
        mem += weight_data_size * sizeof(float);
    }

    if (bias_term)
    {
        bias_data = Mat(num_output, (float*)mem);
        mem += num_output * sizeof(float);
    }

    return 0;
}

int Deconvolution::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    // backward strided convolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

//     fprintf(stderr, "Deconvolution input %d x %d  pad = %d  ksize=%d  stride=%d\n", w, h, pad, kernel_size, stride);

    const int kernel_extent = dilation * (kernel_size - 1) + 1;

#if NCNN_EIGEN
	int _size_in  = w * h;
	int _size_ckk = num_output * kernel_size * kernel_size;
    Mat _top_col(_size_ckk, _size_in);

    typedef Eigen::Matrix<float, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> MatrixRowMajor;
    typedef Eigen::Matrix<float, Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> MatrixColMajor;
    // Eigen::Map<MatrixRowMajor> _weight(weight_data.data,channels,_size_ckk);
    Eigen::Map<MatrixColMajor> _weight(weight_data.data,_size_ckk,channels);
    Eigen::Map<MatrixRowMajor> _bottom(bottom_blob.data,channels,_size_in);
    Eigen::Map<MatrixRowMajor> _top(_top_col.data,_size_ckk,_size_in);
	// _top = _weight.transpose() * _bottom;
	_top = _weight * _bottom;

	// cut
    int outw = (w - 1) * stride + kernel_extent - 2 * pad;
    int outh = (h - 1) * stride + kernel_extent - 2 * pad;
	int _size_out = outw * outh;

    top_blob.create(outw, outh, num_output);
    if (top_blob.empty())
        return -100;

    col2im_cpu<float>(_top_col.data, num_output, outh, outw, kernel_size, kernel_size,
            pad, pad, stride, stride, dilation, dilation, top_blob.data);

	if (bias_term) {
		Eigen::Map<MatrixRowMajor> _top(top_blob.data,num_output,_size_out);
		Eigen::Map<Eigen::VectorXf> _bias(bias_data.data,num_output);
		_top.colwise() += _bias;
	}
#else
    int outw = (w - 1) * stride + kernel_extent;
    int outh = (h - 1) * stride + kernel_extent;

    Mat top_blob_bordered;
    top_blob_bordered.create(outw, outh, num_output);
    if (top_blob_bordered.empty())
        return -100;

    const int maxk = kernel_size * kernel_size;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = outw * dilation - kernel_size * dilation;;
        for (int i = 0; i < kernel_size; i++)
        {
            for (int j = 0; j < kernel_size; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation;
            }
            p2 += gap;
        }
    }

    // num_output
    const float* weight_data_ptr = weight_data;
    #pragma omp parallel for
    for (int p=0; p<num_output; p++)
    {
        Mat out = top_blob_bordered.channel(p);

        const float bias = bias_term ? bias_data.data[p] : 0.f;

        out.fill(bias);

        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                float* outptr = out.data + out.w * i*stride + j*stride;

                const float* kptr = weight_data_ptr + maxk * channels * p;

                // channels
                for (int q=0; q<channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    float val = *(m.data + m.w * i + j);

                    for (int k = 0; k < maxk; k++)
                    {
                        float w = kptr[k];
                        outptr[ space_ofs[k] ] += val * w;
                    }

                    kptr += maxk;
                }
            }
        }
    }

    top_blob = top_blob_bordered;

    if (pad > 0)
    {
        copy_cut_border(top_blob_bordered, top_blob, pad, pad, pad, pad);
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }
#endif

    return 0;
}

} // namespace ncnn
