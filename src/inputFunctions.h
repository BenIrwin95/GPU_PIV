
// function to remove newline characters from a string
void remove_newline(char *str);


// removes all whitespace from a string
void strip_whitespace_inplace(char *str);


// function to return the what follows on the line starting with a keyword
char* find_line_after_keyword(const char* filename, const char* keyword, int* status);

// function to return the integer directly following a specified keyword
int extract_int_by_keyword(const char* filename, const char* keyword, int* status);

// extract a known number of integers from a char array
// returns NULL if the incorrect number of integers is found
int* extract_integer_list_from_char(const char *str, int expected_count, int* status);

// extract a known number of integers from an input file based on a keyword the line starts with
// returns NULL if the incorrect number of integers is found
int* find_int_list_after_keyword(const char* filename, const char* keyword, const int N, int* status);
