
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <vector>

#include "net.h"

int main(int argc, char** argv)
{
    ncnn::Net squeezenet;
    squeezenet.load_param("simple_fc.param");
    squeezenet.load_model("simple_fc.bin");

	float data[3] = {1,1,1};
	ncnn::Mat in(1, 1, 3, data);
	for (int k = 0; k < 3; ++k) {
		printf("%.2f ", in.data[k]);
	}
	printf("\n");

    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.set_light_mode(true);

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("fc1", out);

	std::vector<float> cls_scores;
    cls_scores.resize(out.c);
	std::cout << "out.size=" << out.c << std::endl;
	std::cout << "out.cstep=" << out.cstep << std::endl;
    for (int j=0; j<out.c; j++)
    {
        const float* prob = out.data + out.cstep * j;
        cls_scores[j] = prob[0];
		std::cout << cls_scores[j] << std::endl;
    }

    return 0;
}

