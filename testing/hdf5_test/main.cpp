#include "standardHeader.hpp"
#include <H5Cpp.h>


int main() {
    try {
        // Create a new HDF5 file
        H5::H5File file("groups_official.h5", H5F_ACC_TRUNC);

        // Create a group named "Experiment1"
        H5::Group group1 = file.createGroup("Experiment1");

        // Define a dataspace and datatype for the dataset
        hsize_t dimsf[] = {3};
        H5::DataSpace dataspace(1, dimsf);
        H5::IntType datatype(H5::PredType::NATIVE_INT);

        // Create the dataset inside the group
        std::vector<int> data1 = {1, 2, 3};
        H5::DataSet dataset1 = group1.createDataSet("TrialA", datatype, dataspace);
        dataset1.write(data1.data(), H5::PredType::NATIVE_INT);

        // The objects automatically close when they go out of scope due to RAII
    } catch (const H5::Exception& error) {
        error.printErrorStack();
        return -1;
    }

    std::cout << "Successfully created groups and datasets." << std::endl;
    return 0;
}

// basic example
/*
const std::string FILE_NAME("my_data.h5");
const std::string DATASET_NAME("MyDataset");
const int DIM1 = 5; // Rows
const int DIM2 = 6; // Columns
const int RANK = 2; // Number of dimensions

int main() {
    // Data to be written to the file
    int data[DIM1][DIM2];
    for (int i = 0; i < DIM1; ++i) {
        for (int j = 0; j < DIM2; ++j) {
            data[i][j] = i * DIM2 + j;
        }
    }

    try {
        // Create a new HDF5 file
        H5::H5File file(FILE_NAME, H5F_ACC_TRUNC);

        // Define the dimensions of the dataset
        hsize_t dims[RANK] = {DIM1, DIM2};
        H5::DataSpace dataspace(RANK, dims);

        // Define the data type to use
        H5::IntType datatype(H5::PredType::NATIVE_INT);
        datatype.setOrder(H5T_ORDER_LE);

        // Create the dataset and write the data
        H5::DataSet dataset = file.createDataSet(DATASET_NAME, datatype, dataspace);
        dataset.write(data, H5::PredType::NATIVE_INT);

    } catch (const H5::Exception& error) {
        // Catch any HDF5-specific errors
        error.printErrorStack();
        return -1;
    }

    std::cout << "Data successfully written to " << FILE_NAME << std::endl;

    return 0;
}*/


// writing 1d data to 2D
/*
const std::string FILE_NAME("2d_data_cpp.h5");
const std::string DATASET_NAME("2d_matrix");
const int ROWS = 4;
const int COLS = 5;
const int RANK = 2; // Rank of the dataspace (2 dimensions)

int main() {
    // 1. Create a 1D vector with 20 elements to hold the data
    std::vector<int> data(ROWS * COLS);
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            data[i * COLS + j] = i * 10 + j;
        }
    }

    try {
        // 2. Create the HDF5 file
        H5::H5File file(FILE_NAME, H5F_ACC_TRUNC);

        // 3. Define the dataspace for the 2D data on disk
        hsize_t dims[RANK] = {ROWS, COLS};
        H5::DataSpace dataspace(RANK, dims);

        // 4. Define the data type
        H5::IntType datatype(H5::PredType::NATIVE_INT);
        datatype.setOrder(H5T_ORDER_LE);

        // 5. Create the dataset and write the 1D vector to the 2D dataspace
        H5::DataSet dataset = file.createDataSet(DATASET_NAME, datatype, dataspace);
        dataset.write(data.data(), H5::PredType::NATIVE_INT);

    } catch (const H5::Exception& error) {
        error.printErrorStack();
        return -1;
    }

    std::cout << "2D data successfully written to " << FILE_NAME << std::endl;

    return 0;
}*/
