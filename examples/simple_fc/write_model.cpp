
#include <stdio.h>
#include <stdlib.h>

int main() {
	int quantize_tag = 0;
	float weights[] = {6,7,8,9,10,11,0,1,2,3,4,5};
	FILE *fp = fopen("simple_fc.bin", "w+");
	fwrite(&quantize_tag, sizeof(int), 1, fp);
	fwrite(weights, sizeof(float), sizeof(weights), fp);
	fclose(fp);
}
