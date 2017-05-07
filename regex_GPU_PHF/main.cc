#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "CreateTable/create_PFAC_table_reorder.c"
#include "PHF/phf.c"

#define CUDA_DEVICE_ID 0

int PFAC_table[MAX_STATE][CHAR_SET];  // 2D PFAC state transition table
int num_output[MAX_STATE];            // num of matched pattern for each state
int *outputs[MAX_STATE];              // list of matched pattern for each state

int r[ROW_MAX];          // r[R]=amount row Keys[R][] was shifted
int HT[HASHTABLE_MAX];   // the shifted rows of Keys[][] collapse into HT[]
int val[HASHTABLE_MAX];  // store next state corresponding to hash key, not used in this version

int GPU_TraceTable(unsigned char *input_string, int input_size, int state_num,
    int final_state_num, short *match_result, int HTSize, int width, int *s0Table );

/****************************************************************************
*   Function   : main
*   Description: Main function
*   Parameters : Command line arguments
*   Returned   : Program end success(0) or fail(1)
****************************************************************************/
int main(int argc, char *argv[]) {
    int state_num, final_state_num, type;
    int width, HTSize;
    unsigned char *input_string;
    int input_size;
    short *match_result;
	int i;
	int j;

    
    // check command line arguments
    if (argc != 5) {
        fprintf(stderr, "usage: %s <pattern file name> <type> <PHF width> <input file name>\n", argv[0]);
        exit(-1);
    }
    
    // set which CUDA device to use
    if ( cudaSetDevice(CUDA_DEVICE_ID) != cudaSuccess ) {
        fprintf(stderr, "Set CUDA device %d error\n", CUDA_DEVICE_ID);
        exit(1);
    }
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, CUDA_DEVICE_ID);
    printf("Using Device %d: \"%s\"\n", CUDA_DEVICE_ID, deviceProp.name);
    
    // read pattern file and create PFAC table
    type = atoi(argv[2]);
    create_PFAC_table_reorder(argv[1], &state_num, &final_state_num, type);
    
    printf("state num: %d\n", state_num);
    printf("final state num: %d\n", final_state_num);
	
	//write table to file
    FILE *fw = fopen("PFAC_table.txt", "w");
    if (fw == NULL) {
        perror("Open output file failed.\n");
        exit(1);
    }

    // output PFAC table
    for (i = 0; i < state_num; i++) {
        for (j = 0; j < CHAR_SET; j++) {
            if (PFAC_table[i][j] != -1) {
                fprintf(fw, "state=%2d  '%c'(%02X) ->  %2d\n", i, j, j, PFAC_table[i][j]);
            }
        }
    }
    
    if (type == 2) {
        for (i = 1; i <= final_state_num; i++) {
            fprintf(stdout, "state=%2d  output_pattern= ", i);
			fprintf(fw, "state=%2d  output_pattern= ", i);
            for (j = 0; j < num_output[i]; j++) {
                fprintf(stdout, "%2d ", outputs[i][j]);
				fprintf(fw, "%2d ", outputs[i][j]);
            }
            fprintf(stdout, "\n");
			fprintf(fw, "\n");
        }
    }
    fclose(fw);	
	
	
    // create PHF hash table from PFAC table
    width = atoi(argv[3]);
    HTSize = FFDM((int *)PFAC_table, state_num*CHAR_SET, width);
    
    // read input data
    FILE *fpin = fopen(argv[4], "rb");
    if (fpin == NULL) {
        perror("Open input file failed.");
        exit(1);
    }
    // obtain file size:
    fseek(fpin, 0, SEEK_END);
    input_size = ftell(fpin);
    rewind(fpin);
    
    // allocate host memory: input data
    cudaError_t status;
    status = cudaMallocHost((void **) &input_string, sizeof(char)*input_size);
    if (cudaSuccess != status) {
        fprintf(stderr, "cudaMallocHost input_string error: %s\n", cudaGetErrorString(status));
        exit(1);
    }
    
    // copy the file into the buffer:
    input_size = fread(input_string, sizeof(char), input_size, fpin);
    fclose(fpin);
    
	// allocate host memory: match result
    status = cudaMallocHost((void **) &match_result, sizeof(short)*input_size);
    if (cudaSuccess != status) {
        fprintf(stderr, "cudaMallocHost match_result error: %s\n", cudaGetErrorString(status));
        exit(1);
    }
    
    // exact string matching kernel
	//printf("final_state_num = %d\n",final_state_num);
    GPU_TraceTable(input_string, input_size, state_num, final_state_num,
        match_result, HTSize, width, PFAC_table[(final_state_num+1)] );
	
	// Output results
    FILE *fpout = fopen("GPU_match_result.txt", "w");
    if (fpout == NULL) {
        perror("Open output file failed.\n");
        exit(1);
    }
    // Output match result to file
 	if (type == 0){
		for (i = 0; i < input_size; i++) {
			if (match_result[i] != 0) {
				fprintf(fpout, "At position %4d, match pattern %d\n", i, match_result[i]);
			}
		}
    }
		
		
	if (type == 2) {
		for (i = 0; i < input_size; i++) {
			if (match_result[i] != 0) {
				fprintf(fpout, "At position %4d, match pattern ", i);
				for (j = 0; j < num_output[match_result[i]]; j++) {
					fprintf(fpout, "%2d ", outputs[match_result[i]][j]);
				}
				fprintf(fpout, "\n");
			}
		}		
	}
    fclose(fpout);
	 
			
    
    cudaFreeHost(input_string);
    cudaFreeHost(match_result);
    
    return 0;
}
