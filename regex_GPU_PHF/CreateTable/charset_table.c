#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "def.h"

#define GET_CH     0x200
#define GET_CH_SET 0x400

struct NFAnode {
    int state_id;
    int output_pattern;
    int num_next_state[CHAR_SET];
    struct NFAnode **next_state[CHAR_SET];
};

struct DFAnode {
    int num_state_id;
    int *state_ids;
    int num_output;
    int *output_pattern;
    struct DFAnode *next_state[CHAR_SET];

    int DFA_state_id;
};

typedef struct NFAnode NFA_node;
typedef struct DFAnode DFA_node;

NFA_node *build_NFA(char *patternfilename);
int fgetc_set(FILE *fp, int char_set[CHAR_SET]);
NFA_node *create_NFA_state(void);
int NFA_BFS(NFA_node *init_state);
DFA_node *create_DFA_state(void);
NFA_node *search_NFA_node(NFA_node *init_state, int id);
DFA_node *search_DFA_node(DFA_node *init_state, int num_id, int *ids);
DFA_node *NFA2DFA(NFA_node *NFA_init_state, int *state_num);
int DFA_BFS(DFA_node *init_state);

int NFA_output_table[MAX_STATE] = {0};
extern int PFAC_table[MAX_STATE][CHAR_SET];
extern int output_table[MAX_STATE];


NFA_node *build_NFA(char *patternfilename) {
    int i;
    int state_num = 0;
    int pattern_num = 0;
    int char_set[CHAR_SET];
    NFA_node *NFA_init_state;
    NFA_node *state, *new_state;
    FILE *fpin;
    int ch;
    
    // open input file
    fpin = fopen(patternfilename, "rb");
    if (fpin == NULL) {
        fprintf(stderr, "Open input file \"%s\" failed.\n", patternfilename);
        exit(1);
    }
    
    printf("initial_NFA\n");
    NFA_init_state = create_NFA_state();
    NFA_init_state->state_id = state_num;
    state_num += 1;

    state = NFA_init_state;

    printf("create_NFA\n");
    while (1) {
        ch = fgetc_set(fpin, char_set);

        if ( feof(fpin) ) {
            break;
        }

        if (ch & GET_CH) {
            if ( (ch & 0x1FF) == EOL ) {
                pattern_num += 1;
                state->output_pattern = pattern_num;
                NFA_output_table[state->state_id] = pattern_num;
                state = NFA_init_state;
                continue;
            }
        }

        new_state = create_NFA_state();
        new_state->state_id = state_num;
        state_num += 1;
        
        if (ch & GET_CH) {
            ch = (int)((unsigned char) ch);
            state->num_next_state[ch] += 1;
            state->next_state[ch] = (NFA_node **)realloc( state->next_state[ch], (state->num_next_state[ch])*sizeof(NFA_node*) );
            if ( NULL == state->next_state[ch] ) {
                printf("Reallocate memory error, state_id=%d, ch=%02x\n", state->state_id, ch&0xFF);
                exit(1);
            }
            else {
                (state->next_state[ch])[(state->num_next_state[ch])-1] = new_state;
            }
        }
        else {
            for (i = 0; i < CHAR_SET; i++) {
                if (char_set[i] == 1) {
                    state->num_next_state[i] += 1;
                    state->next_state[i] = (NFA_node **)realloc( state->next_state[i], (state->num_next_state[i])*sizeof(NFA_node*) );
                    if ( NULL == state->next_state[i] ) {
                        printf("Reallocate memory error, state_id=%d, ch=%02x\n", state->state_id, i&0xFF);
                        exit(1);
                    }
                    else {
                        (state->next_state[i])[(state->num_next_state[i])-1] = new_state;
                        //debug:printf("state_id=%d, ch=%c, space=%d\n", state->state_id, i&0xFF, state->num_next_state[i]);
                    }
                }
            }
        }
        
        state = new_state;
    }
    
    fclose(fpin);
    
    return NFA_init_state;
}

int fgetc_set(FILE *fp, int char_set[CHAR_SET]) {
    int i;
    int ch;
    unsigned char ch_l, ch_r;
    int setting = 1;

    for (i = 0; i < CHAR_SET; i++) {
        char_set[i] = 0;
    }
    
    ch = fgetc_ext(fp);
    
    if (ch == '[') {
        ch = fgetc_ext(fp);
        if (ch == '^') {
            for (i = 0; i < CHAR_SET; i++) {
                char_set[i] = 1;
            }
            setting = 0;
            ch = fgetc_ext(fp);
        }
        
        while ( ch != ']' ) {
            if (ch == '-') {
                ch_r = fgetc_ext(fp);
                for (i = ch_l; i <= ch_r; i++)
                    char_set[i] = setting;
            }
            else {
                ch_l = ch;
                char_set[ch_l] = setting;
            }
            
            ch = fgetc_ext(fp);
        }
        
        return GET_CH_SET;
    }
    
    return (int) ((ch & 0x3FF) | GET_CH);
}

NFA_node *create_NFA_state(void) {
    int i;
    NFA_node *snptr;

    snptr = (NFA_node*) malloc(sizeof(NFA_node));
    if (snptr == NULL) {
        printf("create_NFA_state error.\n");
        exit(1);
    }

    for (i = 0; i < CHAR_SET; i++) {
        snptr->num_next_state[i] = 0;
        snptr->next_state[i] = NULL;
    }
    snptr->state_id = -1;
    snptr->output_pattern = 0;

    return snptr;
}

int NFA_BFS(NFA_node *init_state) {
    int i, j;
    NFA_node *queue[4000];
    int queue_head, queue_tail;
    int num_state, sid;
    NFA_node *cur_state;
    int flag[MAX_STATE] = {0};

    queue[0] = init_state;
    queue_head = 0;
    queue_tail = 1;

    while (queue_head != queue_tail) {
        cur_state = queue[queue_head];
        queue_head = (queue_head + 1) % 4000;

        if (flag[cur_state->state_id] != 0)
            continue;
        else
            flag[cur_state->state_id] = 1;

        if (cur_state->output_pattern > 0) {
            fprintf(stdout, "state=%2d  output_pattern=%2d\n", cur_state->state_id, cur_state->output_pattern);
        }

        for (i = 0; i < CHAR_SET; i++) {
            if ( cur_state->num_next_state[i] > 0 ) {
                num_state = cur_state->num_next_state[i];
                for (j = 0; j < num_state; j++) {
                    if (NULL == ((cur_state->next_state[i])[j]) ) printf("NULL_err\n");
                    sid = (cur_state->next_state[i])[j]->state_id;
                    fprintf(stdout, "state=%2d  %c ->  %2d\n", cur_state->state_id, i, sid);
                    queue[queue_tail] = ((cur_state->next_state[i])[j]) ;
                    queue_tail = (queue_tail + 1) % 4000;
                }
            }
        }
    }

    return 0;
}

DFA_node *create_DFA_state(void) {
    int i;
    DFA_node *snptr;

    snptr = (DFA_node*) malloc(sizeof(DFA_node));
    if (snptr == NULL) {
        printf("create_DFA_state error.\n");
        exit(1);
    }

    for (i = 0; i < CHAR_SET; i++) {
        snptr->next_state[i] = NULL;
    }

    snptr->num_state_id = 0;
    snptr->state_ids = NULL;
    snptr->num_output = 0;
    snptr->output_pattern = NULL;
    snptr->DFA_state_id = -1;

    return snptr;
}

NFA_node *search_NFA_node(NFA_node *init_state, int id) {
    int ch, i;
    NFA_node *queue[4000];
    int queue_head, queue_tail;
    int num_state, sid;
    NFA_node *cur_state;

    queue[0] = init_state;
    queue_head = 0;
    queue_tail = 1;

    while (queue_head != queue_tail) {
        cur_state = queue[queue_head];
        if (cur_state->state_id == id) {
            return cur_state;
        }
        queue_head = (queue_head + 1) % 4000;
        for (ch = 0; ch < CHAR_SET; ch++) {
            if ( cur_state->num_next_state[ch] > 0 ) {
                num_state = cur_state->num_next_state[ch];
                for (i = 0; i < num_state; i++) {
                    if (NULL == ((cur_state->next_state[ch])[i]) ) printf("NULL_err\n");
                    sid = (cur_state->next_state[ch])[i]->state_id;
                    //debug:fprintf(stdout, "state=%2d  %c ->  %2d\n", cur_state->state_id, i, sid);
                    queue[queue_tail] = ((cur_state->next_state[ch])[i]) ;
                    queue_tail = (queue_tail + 1) % 4000;
                }
            }
        }
    }

    return NULL;
}

DFA_node *search_DFA_node(DFA_node *init_state, int num_id, int *ids) {
    int ch;
    DFA_node *queue[4000];
    int queue_head, queue_tail;
    DFA_node *cur_state;

    queue[0] = init_state;
    queue_head = 0;
    queue_tail = 1;

    while (queue_head != queue_tail) {
        cur_state = queue[queue_head];
        if (cur_state->num_state_id == num_id) {
            if ( memcmp(cur_state->state_ids, ids, num_id*sizeof(int)) == 0 )
                return cur_state;
        }
        queue_head = (queue_head + 1) % 4000;
        for (ch = 0; ch < CHAR_SET; ch++) {
            if ( cur_state->next_state[ch] != NULL ) {
                queue[queue_tail] = (cur_state->next_state[ch]) ;
                queue_tail = (queue_tail + 1) % 4000;
            }
        }
    }

    return NULL;
}

int comp(const void * a, const void * b) {
  return ( *(int*)a - *(int*)b );
}

DFA_node *NFA2DFA(NFA_node *NFA_init_state, int *state_num) {
    DFA_node *trans_DFA_queue[4000];
    int queue_head, queue_tail;
    DFA_node *cur_DFA_node, *next_DFA_node, *DFA_init_state;
    NFA_node *cur_NFA_node, *next_NFA_node;
    int next_NFA_id;

    int num_active_state;
    int *active_states;
    int ch, i, j, k;

    DFA_init_state = create_DFA_state();
    DFA_init_state->num_state_id = 1;
    DFA_init_state->state_ids = (int *)malloc(sizeof(int));
    DFA_init_state->state_ids[0] = 0;
    DFA_init_state->DFA_state_id = 0;
    *state_num = 1;

    trans_DFA_queue[0] = DFA_init_state;
    queue_head = 0;
    queue_tail = 1;

    while (queue_head != queue_tail) {
        cur_DFA_node = trans_DFA_queue[queue_head];
        queue_head = (queue_head + 1) % 4000;
        //debug: printf("DFA num_state_id: %d\n", cur_DFA_node->num_state_id);

        for (ch = 0; ch < CHAR_SET; ch++) {
            num_active_state = 0;
            active_states = NULL;
            //debug: printf("num_active_state: %d\n", cur_DFA_node->num_state_id);
            //debug: for (i = 0; i < (cur_DFA_node->num_state_id); i++) printf ("%d ", (cur_DFA_node->state_ids)[i]);
            //debug: putchar('\n');
            //debug: printf("ch: %d\n", ch);

            for (i = 0; i < cur_DFA_node->num_state_id; i++) {
                //printf("search NFA id: %d\n", (cur_DFA_node->state_ids)[i]);
                cur_NFA_node = search_NFA_node(NFA_init_state, (cur_DFA_node->state_ids)[i]);
                //debug: if (cur_NFA_node == NULL) printf("NULL error\n");
                //debug: printf("NFA id: %d\n", cur_NFA_node->state_id);

                if (cur_NFA_node->num_next_state[ch] > 0) {
                    for (j = 0; j < (cur_NFA_node->num_next_state[ch]); j++) {
                        next_NFA_node = (cur_NFA_node->next_state[ch])[j];
                        next_NFA_id = next_NFA_node->state_id;

                        //debug: printf("next_NFA_id: %d\n", next_NFA_id);
                        for (k = 0; k < num_active_state; k++) {
                            if (active_states[k] == next_NFA_id) { break; }
                        }

                        if (k == num_active_state) {  // new active state
                            num_active_state += 1;
                            active_states = (int *)realloc(active_states, num_active_state*sizeof(int));
                            if (active_states == NULL) {
                                printf("Reallocate active state error\n");
                                exit(1);
                            }
                            active_states[num_active_state-1] = next_NFA_id;
                        }
                    }
                }
            }

            if (num_active_state > 0) {
                qsort(active_states, num_active_state, sizeof(int), comp);

                //debug: printf("num_active_state: %d\n", num_active_state);
                //debug: for (i = 0; i < num_active_state; i++) printf ("%d ", active_states[i]);
                //debug: putchar('\n');

                next_DFA_node = search_DFA_node(DFA_init_state, num_active_state, active_states);

                if (next_DFA_node != NULL) {
                    cur_DFA_node->next_state[ch] = next_DFA_node;
                }
                else {
                    next_DFA_node = create_DFA_state();
                    next_DFA_node->DFA_state_id = (*state_num)++;
                    next_DFA_node->num_state_id = num_active_state;
                    next_DFA_node->state_ids = (int *)malloc(num_active_state*sizeof(int));
                    memcpy(next_DFA_node->state_ids, active_states, num_active_state*sizeof(int));
                    for (i = 0; i < num_active_state; i++) {
                        if ( NFA_output_table[ active_states[i] ] > 0) {
                            next_DFA_node->num_output += 1;
                            next_DFA_node->output_pattern = (int *)realloc(next_DFA_node->output_pattern, (next_DFA_node->num_output)*sizeof(int));
                            next_DFA_node->output_pattern[(next_DFA_node->num_output)-1] = NFA_output_table[ active_states[i] ];
                        }
                    }

                    cur_DFA_node->next_state[ch] = next_DFA_node;
                }

                trans_DFA_queue[queue_tail] = next_DFA_node;
                queue_tail = (queue_tail + 1) % 4000;
            }

        }
    }

    return DFA_init_state;
}

int DFA_BFS(DFA_node *init_state) {
    int ch, i;
    DFA_node *queue[4000];
    int queue_head, queue_tail;
    int sid;
    DFA_node *cur_state;
    int flag[MAX_STATE] = {0};

    queue[0] = init_state;
    queue_head = 0;
    queue_tail = 1;

    while (queue_head != queue_tail) {
        cur_state = queue[queue_head];
        queue_head = (queue_head + 1) % 4000;

        if (flag[cur_state->DFA_state_id] != 0)
            continue;
        else
            flag[cur_state->DFA_state_id] = 1;
        
        for (i = 0; i < CHAR_SET; i++) {
            PFAC_table[cur_state->DFA_state_id][i] = -1;
        }

        if (cur_state->num_output > 0) {
            fprintf(stdout, "state=%2d  output_pattern=", cur_state->DFA_state_id);
            for (i = 0; i < (cur_state->num_output); i++ )
                fprintf(stdout, "%2d ", cur_state->output_pattern[i]);
            fprintf(stdout, "\n");
            output_table[cur_state->DFA_state_id] = cur_state->output_pattern[0];
        }

        for (ch = 0; ch < CHAR_SET; ch++) {
            if ( cur_state->next_state[ch] != NULL ) {
                sid = (cur_state->next_state[ch])->DFA_state_id;
                fprintf(stdout, "state=%2d  %c ->  %2d\n", cur_state->DFA_state_id, ch, sid);
                queue[queue_tail] = cur_state->next_state[ch] ;
                queue_tail = (queue_tail + 1) % 4000;
                PFAC_table[cur_state->DFA_state_id][ch] = sid;
            }
        }
    }

    return 0;
}
