#include "standardHeader.hpp"



void add_pass_data_to_file(int pass, std::ofstream& outputFile, PIVdata& piv_data, OpenCL_env& env){
    // first pull the necessary data out of the GPU
    const size_t gridSizeBytes = piv_data.arrSize[pass].s[0]*piv_data.arrSize[pass].s[1]*sizeof(float);
    env.queue.enqueueReadBuffer(env.X, CL_TRUE, 0, gridSizeBytes, piv_data.X[pass].data());
    env.queue.enqueueReadBuffer(env.Y, CL_TRUE, 0, gridSizeBytes, piv_data.Y[pass].data());
    env.queue.enqueueReadBuffer(env.U, CL_TRUE, 0, gridSizeBytes, piv_data.U[pass].data());
    env.queue.enqueueReadBuffer(env.V, CL_TRUE, 0, gridSizeBytes, piv_data.V[pass].data());


    outputFile << "Pass " << pass + 1 << " of " << piv_data.N_pass << "\n";
    outputFile << "Window size " << piv_data.window_sizes[pass] << "\n";
    outputFile << "Rows " << piv_data.arrSize[pass].s[1] << "\n";
    outputFile << "Cols " << piv_data.arrSize[pass].s[0] << "\n";
    outputFile << "image_x,image_y,U,V\n";
    outputFile << std::fixed << std::setprecision(2); // Set precision for floats X and Y
    for (int i = 0; i < piv_data.arrSize[pass].s[1]; i++) {
        for (int j = 0; j < piv_data.arrSize[pass].s[0]; j++) {
            int index = i * piv_data.arrSize[pass].s[0] + j;
            outputFile << piv_data.X[pass][index] << "," << piv_data.Y[pass][index] << ",";
            outputFile << std::setprecision(12) << piv_data.U[pass][index] << "," << piv_data.V[pass][index] << "\n";
        }
    }
    outputFile << "\n\n\n\n\n";
}




H5::H5File initialise_output_hdf5(const std::string filename, PIVdata& piv_data){
    H5::H5File file(filename, H5F_ACC_TRUNC);

    // attach some meta-data
    H5::DataSpace attr_dataspace(H5S_SCALAR); // 1. Create a dataspace for the attribute. H5S_SCALAR means a single value.
    H5::IntType attr_datatype(H5::PredType::NATIVE_INT); // 2. Define the integer datatype for the attribute
    H5::Attribute attribute_N_pass = file.createAttribute("N_pass", attr_datatype, attr_dataspace); // 3. Create the attribute on the file
    H5::Attribute attribute_N_frames = file.createAttribute("N_frames", attr_datatype, attr_dataspace);
    // 4. Write the integer value to the attribute
    attribute_N_pass.write(H5::PredType::NATIVE_INT, &piv_data.N_pass);
    attribute_N_frames.write(H5::PredType::NATIVE_INT, &piv_data.N_frames);


    for(int pass=0;pass<piv_data.N_pass;pass++){
        H5::Group pass_Group = file.createGroup(fmt::format("Pass_{}", pass));
        // attach some meta-data
        H5::Attribute attribute_window_size = pass_Group.createAttribute("window_size", attr_datatype, attr_dataspace);
        H5::Attribute attribute_window_overlap = pass_Group.createAttribute("window_overlap", attr_datatype, attr_dataspace);
        attribute_window_size.write(H5::PredType::NATIVE_INT, &piv_data.window_sizes[pass]);
        attribute_window_overlap.write(H5::PredType::NATIVE_INT, &piv_data.window_overlaps[pass]);

        // make datasets for X and Y, which will be common across all frames within a pass
        const int RANK = 2; // Number of dimensions
        const int ROWS = piv_data.arrSize[pass].s[1];
        const int COLS = piv_data.arrSize[pass].s[0];
        hsize_t dims[RANK] = {ROWS, COLS};
        H5::DataSpace dataspace(RANK, dims);
        H5::FloatType datatype(H5::PredType::NATIVE_FLOAT);
        datatype.setOrder(H5T_ORDER_LE); // sets bit order to little endian

        // Create X and Y datasets and write the data
        H5::DataSet X_dataset = pass_Group.createDataSet("X", datatype, dataspace);
        H5::DataSet Y_dataset = pass_Group.createDataSet("Y", datatype, dataspace);
        X_dataset.write(piv_data.X[pass].data(), H5::PredType::NATIVE_FLOAT);
        Y_dataset.write(piv_data.Y[pass].data(), H5::PredType::NATIVE_FLOAT);

        // create groups for U and V, which will store multiple frames
        H5::Group U_Group = pass_Group.createGroup("U");
        H5::Group V_Group = pass_Group.createGroup("V");
        for(int frame=0;frame<piv_data.N_frames;frame++){
            U_Group.createDataSet(fmt::format("frame{:03}", frame), datatype, dataspace);
            V_Group.createDataSet(fmt::format("frame{:03}", frame), datatype, dataspace);
        }
    }

    return file;
}

/*
// start by making groups for X and Y, which will be common across all frames
H5::Group XGroup = file.createGroup("X");
H5::Group YGroup = file.createGroup("Y");
for(int pass=0;pass<piv_data.N_pass;pass++){
    const int RANK = 2; // Number of dimensions
    const int ROWS = piv_data.arrSize[pass].s[1];
    const int COLS = piv_data.arrSize[pass].s[0];
    hsize_t dims[RANK] = {ROWS, COLS};
    H5::DataSpace dataspace(RANK, dims);
    H5::FloatType datatype(H5::PredType::NATIVE_FLOAT);
    datatype.setOrder(H5T_ORDER_LE); // sets bit order to little endian

    // Create the dataset and write the data
    H5::DataSet X_dataset = XGroup.createDataSet(fmt::format("Pass_{}", pass), datatype, dataspace);
    H5::DataSet Y_dataset = YGroup.createDataSet(fmt::format("Pass_{}", pass), datatype, dataspace);
    X_dataset.write(piv_data.X[pass].data(), H5::PredType::NATIVE_FLOAT);
    Y_dataset.write(piv_data.Y[pass].data(), H5::PredType::NATIVE_FLOAT);
}*/
