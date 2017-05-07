#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROW_MAX        32768
#define COL_MAX        4096
#define HASHTABLE_MAX  163840

#define SHOW_PHF


struct RowStruct {    // structure for each row of Keys[][]
    int RowNumber;    // the row number in array Keys[][]
    int RowItemCnt;   // the # of items in this row of Keys[][]
    int *RowItemIdx;  // the column index of all items in this row
};

// the arrays Keys[][], r[] and HT[] are those in the article "Perfect Hashing"
int Keys[ROW_MAX][COL_MAX];     // Keys[i][j]=K (i=K/w, j=K mod w) for each key K
extern int r[ROW_MAX];          // r[R]=amount row Keys[R][] was shifted
extern int HT[HASHTABLE_MAX];   // the shifted rows of Keys[][] collapse into HT[]

extern int val[HASHTABLE_MAX];  // store next state corresponding to hash key, not used in this version

// Row[] exists to facilitate sorting the rows of Keys[][] by their "fullness"
struct RowStruct Row[ROW_MAX];  // entry counts for the rows in Keys[][]


/****************************************************************************
*   Function   : PHF
*   Description: This function search queried key in hash table. It is not 
*                used now.
*   Parameters : queryKey - Queried key number
*                width - The width of key table
*                HTSize - Size of hash table
*   Returned   : Value corresponding to queried key or -1 (not in hash table)
****************************************************************************/
int PHF(int queryKey, int width, int HTSize) {
    int row, col, index;
    
    row = queryKey / width;
    col = queryKey % width;
    index = r[row] + col;
	// In HT[], right 15 bits record row of key, left 17 bits record key number
    if (index < HTSize && (HT[index] & 0x7FFF) == row) {
        return (HT[index] >> 15) & 0x1FFFF;
    }
    
    return -1;
}

/****************************************************************************
*   Function   : InitArrays
*   Description: This function initialize all arrays.
*   Parameters : None
*   Returned   : No use
****************************************************************************/
void InitArrays(void) {
    int row;
    
	// initialize following arrays to -1
    memset ( Keys, 0xFF, ROW_MAX*COL_MAX*sizeof(int) );    
    memset ( r, 0xFF, ROW_MAX*sizeof(int) );
    memset ( HT, 0xFF, HASHTABLE_MAX*sizeof(int) );
    memset ( val, 0xFF, HASHTABLE_MAX*sizeof(int) );
    
    for (row = 0; row < ROW_MAX; row++) {
        Row[row].RowNumber  = row;   // insert the row numbers and
        Row[row].RowItemCnt = 0;     // indicate that each row is empty
        Row[row].RowItemIdx = NULL;  // the item index in the row
    }
}

/****************************************************************************
*   Function   : ReadKey
*   Description: This function reads data in ary[] (PFAC transition table)
*                to fill Key[][] (key table) and Row[] (row structure)
*   Parameters : ary - Key array, corresponding to PFAC table
*                ary_size - Size of ary
*                width - The width of key table
*                KeyCount - Address of varible to store total number of keys
*                MaxKey - Address of varible to store max key number
*   Returned   : No use
****************************************************************************/
int ReadKey(int *ary, int ary_size, int width, int *KeyCount, int *MaxKey) {
    int key;
    int row, col;
    
    *KeyCount = 0;  // set # keys=0 before attempting fopen
    *MaxKey = 0;
    
    // fill data structures using key data
    for (key = 0; key < ary_size; key++) {
        if (ary[key] < 0) continue;
        row = key / width;
        col = key % width;
        if (row >= ROW_MAX) {
            fprintf(stderr, "Row > ROW_MAX(%d)\n", ROW_MAX);
            exit(-3);
        }
        Keys[row][col] = key;
        Row[row].RowItemCnt += 1;
        Row[row].RowItemIdx = (int *) realloc(Row[row].RowItemIdx, Row[row].RowItemCnt*sizeof(int));
        Row[row].RowItemIdx[ (Row[row].RowItemCnt)-1 ] = col;
        (*KeyCount) += 1;
        if (key > *MaxKey) {
            *MaxKey = key;
        }
    }
	
    return 0;
}

/****************************************************************************
*   Function   : SortRows
*   Description: This function sort Row[] according to the number of keys in
*                a row
*   Parameters : numRow - Key array, corresponding to PFAC table
*   Returned   : None
****************************************************************************/
void SortRows(int numRow) {
    int i, j;
    struct RowStruct tmp;
    
    for (i = 0; i < numRow-1; i++) {
        for (j = i+1; j < numRow; j++) {
            if (Row[i].RowItemCnt < Row[j].RowItemCnt) {
                tmp    = Row[i];
                Row[i] = Row[j];
                Row[j] = tmp;
            }
        }
    }
}

/****************************************************************************
*   Function   : FFDM (First-Fit Descending Method algorithm)
*   Description: This function create PHF. Store row of key in right 15 bits 
*                of HT[] for verification, and store data in ary[] (next 
*                state) in left 17 bits of HT[].
*   Parameters : ary - Key array, corresponding to PFAC table
*                ary_size - Size of ary
*                width - The width of key table
*   Returned   : Size of hash table
****************************************************************************/
int FFDM(int *ary, int ary_size, int width) {
    int NumKeys, MaxKey, MaxOffset, MaxRow, HTSize = 0;
    int i, ndx, rc, row, col, key, mergeVal;
    int offset, rowItemCnt;
    int *rowPtr;
    
    if (width > COL_MAX) {
        printf("width may not exceed %d\n", COL_MAX);
        exit(-1);
    }
    
    InitArrays();
    
    // read in the user's key data
    rc = ReadKey(ary, ary_size, width, &NumKeys, &MaxKey);
    if (rc != 0) {
        printf("ReadKey() failed with error %d\n", rc);
        exit(rc);
    }
    
    // prime the algorithm - sort the rows by their fullness
    MaxRow = MaxKey / width + 1;
    SortRows( MaxRow );
    
    // do the First-Fit Descending Method algorithm
    // For each non-empty row:
    // 1. shift the row right until none of its items collide with any of
    //    the items in previous rows.
    // 2. Record the shift amount in array r[].
    // 3. Insert this row into the hash table HT[].
    
    MaxOffset = 0;
    for (ndx = 0; Row[ndx].RowItemCnt > 0; ndx++) {
        row = Row[ndx].RowNumber;      // get the next non-empty row
        rowItemCnt = Row[ndx].RowItemCnt;
        rowPtr = Row[ndx].RowItemIdx;
        for (offset = -1*rowPtr[0]; offset < HASHTABLE_MAX-width; offset++) {
            for (i = 0; i < rowItemCnt; i++) {
                col = rowPtr[i];
                if (HT[offset+col] != -1) {
                    break;
                }
            }
            
            if (i == rowItemCnt) {
                r[row] = offset;      // record the shift amount for this row
                if (offset > MaxOffset) {
                    MaxOffset = offset;
                }
                for (i = 0; i < rowItemCnt; i++) {  // insert this row into the hash table
                    col = rowPtr[i];
                    key = row * width + col;
                    mergeVal = 0;
                    mergeVal |= row;
                    mergeVal |= (ary[key] << 15);
                    HT[offset+col] = mergeVal;
                    // HT[offset+col] = row;
                    // val[offset+col] = ary[key];
                }
                break;
            }
        }
        
        if (offset == HASHTABLE_MAX-width) {
            printf("failed to fit row %d into the hash table\n", row);
            printf("try increasing the hash table size\n");
            exit(-1);
        }
    }
    
    // compute hash table size
    for (i = MaxOffset; i < MaxOffset + width; i++) {
        if (HT[i] > 0) {
            HTSize = i + 1;
        }
    }
    
#ifdef SHOW_PHF
    // print the results
    printf("\n");
    printf("Number of keys    : %d\n", NumKeys );
    printf("Max Key           : %d\n", MaxKey );
    printf("width value       : %d\n", width );
    printf("\n");
    printf("Max Offset        : %d\n", MaxOffset );
    printf("r table size      : \033[93m%7d\033[m", MaxRow );
    printf("  (\033[93m%7lu\033[m Bytes)\n", MaxRow * sizeof(int) );
    printf("Hash table size   : \033[96m%7d\033[m", HTSize );
    printf("  (\033[96m%7lu\033[m Bytes)\n", (HTSize) * sizeof(int) );
    printf("Memory size       : \033[97m%7lu\033[m Bytes\n", (MaxRow + HTSize) * sizeof(int) );
    printf("Table utilization : \033[92m%.3f %%\033[m\n", 100.0f*NumKeys/HTSize );
    printf("(\033[4mNumber of keys\033[m / \033[4mHash table size\033[m)\n");
    printf("Table size ratio  : \033[95m%.3f %%\033[m\n", 100.0f*(MaxRow+HTSize)/(MaxKey+256-MaxKey%256) );
    printf("((\033[4mr table size\033[m + \033[4mHash table size\033[m) / \033[4m2D PFAC table size\033[m)\n");
    printf("\n");
#endif
    
    return HTSize;
}
