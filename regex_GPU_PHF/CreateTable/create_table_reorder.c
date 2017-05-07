#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ctdef.h"

// pattern structure
struct ps {
    int pattern_id;
    int pattern_len;
    char *pat;
};

typedef struct ps pattern_s;

pattern_s all_pattern[MAX_STATE];
int pattern_num;


/****************************************************************************
*   Function   : comp_pat
*   Description: This function compare 2 pattern structure for qsort()
*   Parameters : a - 1st pattern
*                b - 2nd pattern
*   Returned   : Comparison result
****************************************************************************/
int comp_pat(const void *a, const void*b) {
    pattern_s *pat1 = (pattern_s *)a;
    pattern_s *pat2 = (pattern_s *)b;
    int str_len1;
    int str_len2;
    int min_len;
    int result;
    
    str_len1 = pat1->pattern_len;
    str_len2 = pat2->pattern_len;
    min_len = (str_len1 < str_len2) ? str_len1 : str_len2;
    
    result = memcmp(pat1->pat , pat2->pat, min_len);
    
    if (result == 0) {
        if (str_len1 > str_len2)
            return 1;
        else if (str_len1 < str_len2)
            return -1;
        else
            return 0;
    }
    
    return result;
}

/****************************************************************************
*   Function   : read_pattern
*   Description: This function read patterns from file to all_pattern[]
*   Parameters : patternfilename - Pattern file name string
*   Returned   : No use
****************************************************************************/
int read_pattern(char *patternfilename) {
    int ch;
    char str[1024];  // pattern length must less than 1024 in PFAC algo
    int str_len;
    FILE *fpin;
    
    pattern_num = 0;
    
    // open input file
    fpin = fopen(patternfilename, "rb");
    if (fpin == NULL) {
        perror("Open input file failed.");
        exit(1);
    }
    
    while (1) {
        // read a pattern
        str_len = 0;
        while (1) {
            ch = fgetc(fpin);
            str[str_len++] = ch;
            
            if (str_len >= 1024) {
                printf("Pattern %d length over 1024.\n", pattern_num+1);
                exit(1);
            }
            
            if (ch == '\n') {
                str_len -= 1;
                pattern_num += 1;
                break;
            }
        }
        
        // put the read pattern to all_pattern[]
        all_pattern[pattern_num].pattern_id = pattern_num;
        all_pattern[pattern_num].pattern_len = str_len;
        all_pattern[pattern_num].pat = (char *)malloc( str_len*sizeof(char) );
        memcpy(all_pattern[pattern_num].pat, str, str_len*sizeof(char));
        
        // check end-of-file
        ch = fgetc(fpin);
        if ( feof(fpin) ) {
            break;
        }
        else {
            ungetc(ch, fpin);
        }
    }
    
    // sort the patterns for correctness of creating table
    qsort(&all_pattern[1], pattern_num, sizeof(pattern_s), comp_pat); 
    
    fclose(fpin);
    
    return 0;
}

/****************************************************************************
*   Function   : read_pattern_ext
*   Description: This function read patterns from file to all_pattern[]
                 using fgetc_ext() instead of fgetc()
*   Parameters : patternfilename - Pattern file name string
*   Returned   : No use
****************************************************************************/
int read_pattern_ext(char *patternfilename) {
    int ch;
    char str[1024];  // pattern length must less than 1024 in PFAC algo
    int str_len;
    FILE *fpin;
    
    pattern_num = 0;
    
    // open input file
    fpin = fopen(patternfilename, "rb");
    if (fpin == NULL) {
        perror("Open input file failed.");
        exit(1);
    }
    
    while (1) {
        // read a pattern
        str_len = 0;
        while (1) {
            ch = fgetc_ext(fpin);
            str[str_len++] = ch;
            
            if (str_len >= 1024) {
                printf("Pattern %d length over 1024.\n", pattern_num+1);
                exit(1);
            }
            
            if (ch == EOL) {
                str_len -= 1;
                pattern_num += 1;
                break;
            }
        }
        
        // put the read pattern to all_pattern[]
        all_pattern[pattern_num].pattern_id = pattern_num;
        all_pattern[pattern_num].pattern_len = str_len;
        all_pattern[pattern_num].pat = (char *)malloc( str_len*sizeof(char) );
        memcpy(all_pattern[pattern_num].pat, str, str_len*sizeof(char));
        
        // check end-of-file
        ch = fgetc(fpin);
        if ( feof(fpin) ) {
            break;
        }
        else {
            ungetc(ch, fpin);
        }
    }
    
    // sort the patterns for correctness of creating table
    qsort(&all_pattern[1], pattern_num, sizeof(pattern_s), comp_pat); 
    
    fclose(fpin);
    
    return 0;
}

/****************************************************************************
*   Function   : create_table_reorder
*   Description: create transition table from all_pattern[]
*   Parameters : patternfilename - Pattern file name string
*                state_num - Address of variable to store total number of
*                            state
*                final_state_num - Address of variable to store total number
*                                  of final state
*                ext - Extension mode selection. 0 for normal ASCII and
                       1 for reading escape character
*   Returned   : No use (total number of state)
****************************************************************************/
int create_table_reorder(char *patternfilename, int *state_num, int *final_state_num, int ext) {
    int i, j;
    int ch;
    int state;          // to traverse transition table
    int state_count;    // counter for creating new state
    int initial_state;
    pattern_s cur_pat;
    
    // initialize transition table
    for (i = 0; i < MAX_STATE; i++) {
        for (j = 0; j < CHAR_SET; j++) {
            PFAC_table[i][j] = -1;
        }
    }
    
    // select normal mode or extension mode
    if (ext == 0)
        read_pattern(patternfilename);
    else
        read_pattern_ext(patternfilename);
    
    // final states are state[1] ~ state[n], n is number of pattern
    *final_state_num = pattern_num;
    initial_state = *final_state_num + 1;
    // state start from initial state
    state = initial_state;
    // create new state from (initial_state+1)
    state_count = initial_state + 1;
    
    // traverse all_pattern[] and create the transition table
    for (i = 1; i <= pattern_num; i++) {
        // load current pattern
        cur_pat = all_pattern[i];
        
        // create transition according to pattern
        for (j = 0; j < cur_pat.pattern_len-1; j++) {
            ch = (unsigned char)cur_pat.pat[j];
            
            if (PFAC_table[state][ch] == -1) {
                PFAC_table[state][ch] = state_count;
                state = state_count;
                state_count += 1;
            }
            else {
                state = PFAC_table[state][ch];
            }
        }
        
        // the ending char will create a transition to corresponding final state
        ch = (unsigned char)cur_pat.pat[j];
        PFAC_table[state][ch] = cur_pat.pattern_id;
        // initialize state to load next pattern
        state = initial_state;
        
        // check state overflow
        if (state_count > MAX_STATE) {
            fprintf(stderr, "State number overflow, %d\n", state_count);
            exit(1);
        }
    }
    
    *state_num = state_count;
    
    return state_count;
}
