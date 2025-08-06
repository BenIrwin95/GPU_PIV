#include "standardHeader.hpp"



int findIntegerAfterKeyword(const std::string& filename, const std::string& keyword) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file '" + filename + "'");
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind(keyword, 0) == 0) {
            std::stringstream ss(line.substr(keyword.length()));
            int number;

            ss >> number;

            if (!ss.fail()) {
                file.close();
                return number;
            } else {
                file.close();
                throw std::runtime_error("Error: Found keyword '" + keyword + "' but could not parse an integer.");
            }
        }
    }

    file.close();
    throw std::runtime_error("Error: Keyword '" + keyword + "' not found in file '" + filename + "'");
}


std::string findRestOfLineAfterKeyword(const std::string& filename, const std::string& keyword) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file '" + filename + "'");
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind(keyword, 0) == 0) {
            // Keyword found. Extract the rest of the line.
            std::string result = line.substr(keyword.length());

            // Trim leading whitespace
            size_t first = result.find_first_not_of(" \t\n\r");
            if (std::string::npos == first) {
                file.close();
                return ""; // The rest of the line was entirely whitespace
            }
            // Trim trailing whitespace
            size_t last = result.find_last_not_of(" \t\n\r");

            file.close();
            return result.substr(first, (last - first + 1));
        }
    }

    file.close();
    throw std::runtime_error("Error: Keyword '" + keyword + "' not found in file '" + filename + "'");
}


std::vector<int> findIntegersAfterKeyword(const std::string& filename, const std::string& keyword) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file '" + filename + "'");
    }

    std::string line;
    while (std::getline(file, line)) {
        // Check if the line starts with the keyword, followed by a space
        if (line.rfind(keyword, 0) == 0 && line.length() > keyword.length() && line[keyword.length()] == ' ') {
            // Keyword found. Extract the rest of the line after the keyword and the space.
            std::string rest_of_line = line.substr(keyword.length() + 1);

            std::stringstream ss(rest_of_line);
            std::vector<int> integers;
            int value;

            while (ss >> value) {
                integers.push_back(value);
            }

            file.close();
            return integers;
        }
    }

    file.close();
    throw std::runtime_error("Error: Keyword '" + keyword + "' not found in file '" + filename + "'");
}
