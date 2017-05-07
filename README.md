# Perfect-hashing-based-PFAC
The file structure is as following.
```
main.cc                          // main function
+-CreateTable/
| +-create_PFAC_table_reorder.c    // package 3 kinds of creating table method
|   +-create_table_reorder.c       // read pattern as normal text or with escape character
|   | +-ctdef.h                    // create table definitions, and fgetc_ext()
|   |
|   +-charset_table_reorder.c      // read pattern as char set representation
|     +-ctdef.h
|
+-PHF/
  +-phf.c                          // create perfect hash function
```
Use make to compile. The executable file is named as 'gphf'. 

usage:
```
./gphf <pattern file name> <type> <PHF width> <input file name>
```
type:
```
  0  normal text
  1  text with escape character
  2  character set representation
```
