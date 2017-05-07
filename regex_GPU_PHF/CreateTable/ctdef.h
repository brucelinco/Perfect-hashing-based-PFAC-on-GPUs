#ifndef _DEF_H
#define _DEF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_STATE  131072  // max number of state in 17 bits
#define CHAR_SET   256     // ASCII character set
#define EOL        0x10A   // merge '\' and 'n' into escape char '\n'(LF)
                           // differentiate EOL (separate patterns) from merged LF

extern int PFAC_table[MAX_STATE][CHAR_SET];  // 2D PFAC state transition table
extern int num_output[MAX_STATE];            // num of matched pattern for each state
extern int *outputs[MAX_STATE];              // list of matched pattern for each state

/****************************************************************************
*   Function   : fgetc_ext
*   Description: This function is an extension of fgetc. It merge '\' and 
*                some char to escape character and return its ASCII code.
*   Parameters : fp - File pointer to be read
*   Returned   : ASCII code of a char or EOL
****************************************************************************/
int fgetc_ext(FILE *fp) {
    int ch[2];
    int value = 0;
    int ret;
    
    ch[0] = fgetc(fp);
    
    // recognize '\' as start of escape character
    if (ch[0] == '\\') {
        ch[1] = fgetc(fp);
        
        // for ch[1] is EOF
        if ( feof(fp) ) {
            return ch[0];
        }
        
        // for '\ooo' octal representation
        if ( isdigit(ch[1]) ) {
            ungetc(ch[1], fp);
            fscanf(fp, "%3o", &value);
            return ((int)((char)value));
        }
        
        switch (ch[1]) {
            case 'a':
                return '\a';
            case 'b':
                return '\b';
            case 't':
                return '\t';
            case 'n':
                return '\n';
            case 'v':
                return '\v';
            case 'f':
                return '\f';
            case 'r':
                return '\r';
            case '\'':
            case '\"':
            case '\\':
                return ch[1];
            case 'x':
                // for '\xnn' hexadecimal representation
                ret = fscanf(fp, "%2x", &value);
                if (ret == 0) {
                    fprintf(stderr, "Syntax error: \\x used with no following hex digits\n");
                }
                return ((int)((char)value));
            default:
                // '\' and ch[1] are not escape character
                ungetc(ch[1], fp);
                return ch[0];
        }
    }
    
    // for separating patterns
    if (ch[0] == '\n') {
        return EOL;
    }
    
    return ch[0];
}

#endif
