#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "create_table_reorder.c"
#include "charset_table_reorder.c"

int create_PFAC_table_reorder(char *patternfilename, int *state_num, int *final_state_num, int type) {
    NFA_node *NFA_init_state = NULL;
    DFA_node *DFA_init_state = NULL;
    
    if (type >= 0 && type <= 1) {
        create_table_reorder(patternfilename, state_num, final_state_num, type);
    }
    else {
        // printf("Build NFA\n");
        NFA_init_state = build_NFA(patternfilename);
        
        // printf("NFA_BFS\n");
        // NFA_BFS(NFA_init_state);
        
        DFA_init_state = NFA2DFA(NFA_init_state, state_num, final_state_num);
        
        mark_DFA_id(DFA_init_state, *final_state_num);
        
        // printf("DFA_BFS\n");
        DFA_BFS(DFA_init_state);
    }
    
    return 0;
}