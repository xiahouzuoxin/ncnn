#pragma once
#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "layer/deconvolution.h"
using namespace ncnn;


TEST(deconvolution, forward) 
{
    // layer params
    Deconvolution deconv_layer;
    deconv_layer.num_output = 2;
    deconv_layer.kernel_size = 3;
    deconv_layer.dilation = 1;
    deconv_layer.stride = 2;
    deconv_layer.pad = 0;
    deconv_layer.bias_term = 1;
    deconv_layer.weight_data_size = 9;

    // input & output
    float_t in[] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f
    };

    float_t expected_out[] = {
		0.5, 0.5,   1,   1,   2, 1.5,    3,   2,    4, 2.5, 2.5,
		0.5, 0.5,   1,   1,   2, 1.5,    3,   2,    4, 2.5, 2.5,
		  1,   1, 2.5,   2, 4.5,   3,  6.5,   4,  8.5,   5,   5,
		  1,   1,   2, 1.5,   3,   2,    4, 2.5,    5,   3,   3,
		  2,   2, 4.5,   3, 6.5,   4,  8.5,   5, 10.5,   6,   6,
		1.5, 1.5,   3,   2,   4,  2.5,   5,   3,    6, 3.5, 3.5,
		  3,   3, 6.5,   4, 8.5,   5, 10.5,   6, 12.5,   7,   7,
		  2,   2,   4, 2.5,   5,   3,    6, 3.5,    7,   4,   4,
		  4,   4, 8.5,   5,10.5,   6, 12.5,   7, 14.5,   8,   8,
		2.5, 2.5,   5,   3,   6, 3.5,    7,   4,    8, 4.5, 4.5,
		2.5, 2.5,   5,   3,   6, 3.5,    7,   4,    8, 4.5, 4.5,

		0.5, 0.5,   1,   1,   2, 1.5,    3,   2,    4, 2.5, 2.5,
		0.5, 0.5,   1,   1,   2, 1.5,    3,   2,    4, 2.5, 2.5,
		  1,   1, 2.5,   2, 4.5,   3,  6.5,   4,  8.5,   5,   5,
		  1,   1,   2, 1.5,   3,   2,    4, 2.5,    5,   3,   3,
		  2,   2, 4.5,   3, 6.5,   4,  8.5,   5, 10.5,   6,   6,
		1.5, 1.5,   3,   2,   4,  2.5,   5,   3,    6, 3.5, 3.5,
		  3,   3, 6.5,   4, 8.5,   5, 10.5,   6, 12.5,   7,   7,
		  2,   2,   4, 2.5,   5,   3,    6, 3.5,    7,   4,   4,
		  4,   4, 8.5,   5,10.5,   6, 12.5,   7, 14.5,   8,   8,
		2.5, 2.5,   5,   3,   6, 3.5,    7,   4,    8, 4.5, 4.5,
		2.5, 2.5,   5,   3,   6, 3.5,    7,   4,    8, 4.5, 4.5 
    };


    // weights & bias
    float_t w[] = {
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f
    };

    float_t b[] = {
        0.5f,
		0.5f
    };
 
    // forward
    Mat mat_in(5, 5, 1, in);
    Mat mat_out;

    deconv_layer.bias_data.data = b;
    deconv_layer.weight_data.data = w;
    deconv_layer.forward(mat_in, mat_out);

#if 1
    // check expect
    EXPECT_EQ(mat_out.w, 11);
    EXPECT_EQ(mat_out.h, 11);
    EXPECT_EQ(mat_out.c, 2);
    for (int i = 0; i < sizeof(expected_out)/sizeof(expected_out[0]); ++i)
    {
        EXPECT_NEAR(mat_out[i], expected_out[i], 1E-5);
    }
#else
	for (int r = 0; r < mat_out.h; ++r) {
		const float* ptr_row = mat_out.row(r);
		for (int c = 0; c < mat_out.w; ++c) {
			std::cout << ptr_row[c] << " ";
		}
		std::cout << std::endl;
	}
#endif

}

