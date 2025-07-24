#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>





void remove_newline(char *str) {
    if (str == NULL) {
        return;
    }

    size_t len = strlen(str);
    if (len > 0 && str[len - 1] == '\n') {
        str[len - 1] = '\0';
    }
}



void strip_whitespace_inplace(char *str) {
    if (str == NULL) {
        return;
    }

    int i, j = 0;
    for (i = 0; str[i] != '\0'; i++) {
        if (!isspace((unsigned char)str[i])) {
            str[j++] = str[i];
        }
    }
    str[j] = '\0'; // Null-terminate the stripped string
}


char* find_line_after_keyword(const char* filename, const char* keyword, int* status) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("ERROR: could not open file '%s'\n",filename);
        *status = 1;
        return NULL;
    }

    char* line = NULL;
    size_t len = 0;
    ssize_t read;
    size_t keyword_len = strlen(keyword);

    while ((read = getline(&line, &len, file)) != -1) {
        // Check if the line starts with the keyword
        if (strncmp(line, keyword, keyword_len) == 0) {
            // Calculate the starting position of the content after the keyword
            size_t content_start = keyword_len;

            // Skip any leading whitespace after the keyword
            while (content_start < read && (line[content_start] == ' ' || line[content_start] == '\t')) {
                content_start++;
            }

            // Allocate memory for the rest of the line
            size_t remaining_len = read - content_start;
            char* result = (char*)malloc(remaining_len + 1);
            if (result == NULL) {
                printf("ERROR: could not allocate memory for %s\n", keyword);
                *status = 1;
                fclose(file);
                free(line);
                return NULL;
            }

            // Copy the rest of the line
            strncpy(result, line + content_start, remaining_len);
            result[remaining_len] = '\0'; // Null-terminate the result

            fclose(file);
            free(line);
            remove_newline(result);
            return result; // Return the allocated string
        }
    }

    // Keyword not found in any line
    printf("ERROR: Keyword '%s' not found\n",keyword);
    *status = 1;
    fclose(file);
    free(line);
    return NULL;
}



int extract_int_by_keyword(const char* filename, const char* keyword, int* status){
    char* extracted_line = find_line_after_keyword(filename, keyword, status);
    if(*status != 1){
        strip_whitespace_inplace(extracted_line);
        int output = atoi(extracted_line);
        free(extracted_line);
        return output;
    } else {
        return 0;
    }
}



int* extract_integer_list_from_char(const char *str, int expected_count, int* status) {
    // Handle NULL or empty input string, or invalid expected_count
    if (str == NULL || strlen(str) == 0 || expected_count <= 0) {
        *status=1;
        return NULL;
    }

    // Create a mutable copy of the input string because strtok modifies the string.
    char *str_copy = strdup(str);
    if (str_copy == NULL) {
        printf("Failed to duplicate string\n");
        *status=1;
        return NULL; // Memory allocation failed
    }

    // Allocate memory for the expected number of integers upfront
    int *integers = (int *)malloc(expected_count * sizeof(int));
    if (integers == NULL) {
        printf("Failed to allocate memory for integers\n");
        free(str_copy); // Free the string copy
        *status=1;
        return NULL;    // Memory allocation failed
    }

    // Tokenize the string using whitespace as delimiters
    char *token = strtok(str_copy, " \t\n\r\f\v"); // Standard whitespace characters

    int current_extracted = 0; // Counter for integers actually extracted

    while (token != NULL && current_extracted < expected_count) {
        // Convert the token string to an integer and store it
        integers[current_extracted] = atoi(token);
        current_extracted++; // Increment the count of extracted integers

        // Get the next token
        token = strtok(NULL, " \t\n\r\f\v");
    }

    // Free the duplicated string
    free(str_copy);

    // Check if the actual number of extracted integers matches the expected count
    if (current_extracted != expected_count) {
        printf("Error: Expected %d integers but found %d.\n", expected_count, current_extracted);
        free(integers); // Free the allocated memory as the count doesn't match
        *status=1;
        return NULL;    // Return NULL to indicate an error
    }

    return integers;
}



int* find_int_list_after_keyword(const char* filename, const char* keyword, const int N, int* status){
    char * temp = find_line_after_keyword(filename, keyword,status);
    int* output = extract_integer_list_from_char(temp, N, status);
    if(*status == 1){
        printf("ERROR: Something went wrong extracting the integers after keyword %s\n", keyword);
    }
    free(temp);
    return output;

}
