import numpy as np
import matplotlib.pyplot as plt

def unpack_data(filename):
    """
    Unpacks data from a text file, structured by "Pass" sections.

    The file is expected to have sections starting with "Pass",
    followed by "Rows: <num_rows>", "Cols: <num_cols>", and
    then a header line "image_x,image_y,U,V" followed by data.

    Args:
        filename (str): The path to the input text file.

    Returns:
        dict: A dictionary containing the unpacked data with keys:
              - 'N_pass': Number of data passes found.
              - 'X': A list of NumPy arrays, where each array is the X grid for a pass.
              - 'Y': A list of NumPy arrays, where each array is the Y grid for a pass.
              - 'U': A list of NumPy arrays, where each array is the U velocity component grid for a pass.
              - 'V': A list of NumPy arrays, where each array is the V velocity component grid for a pass.
              Returns None if the file cannot be read or no "Pass" sections are found.
    """
    data = {
        'N_pass': 0,
        'X': [],
        'Y': [],
        'U': [],
        'V': []
    }

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        return None

    # Find the starting line for each "Pass" section
    flagged_lines = []
    for i, line in enumerate(lines):
        if line.strip().startswith("Pass"):
            flagged_lines.append(i)

    if not flagged_lines:
        print("No 'Pass' sections found in the file.")
        return data # Return empty data structure

    data['N_pass'] = len(flagged_lines)

    current_line_idx = 0
    for n in range(data['N_pass']):
        N_rows = 0
        N_cols = 0
        data_start_line_idx = -1

        # Start searching from the flagged line for the current pass
        # This ensures we only process relevant lines for the current pass
        current_line_idx = flagged_lines[n]

        # Determine the end of the search for this pass
        # If it's not the last pass, search until the next flagged_line
        # Otherwise, search until the end of the file
        end_search_idx = flagged_lines[n+1] if n + 1 < data['N_pass'] else len(lines)

        while current_line_idx < end_search_idx:
            line = lines[current_line_idx].strip()

            if line.startswith("Rows"):
                try:
                    N_rows = int(line.split(" ")[-1])
                except ValueError:
                    print(f"Warning: Could not parse Rows in line {current_line_idx + 1}. Skipping pass {n+1}.")
                    break # Break out of inner loop to skip this pass
            elif line.startswith("Cols"):
                try:
                    N_cols = int(line.split(" ")[-1])
                except ValueError:
                    print(f"Warning: Could not parse Cols in line {current_line_idx + 1}. Skipping pass {n+1}.")
                    break # Break out of inner loop to skip this pass
            elif line.startswith("image_x,image_y,U,V"):
                data_start_line_idx = current_line_idx + 1
                break # Found header, break from inner loop

            current_line_idx += 1

        if N_rows == 0 or N_cols == 0 or data_start_line_idx == -1:
            print(f"Warning: Missing 'Rows', 'Cols', or data header for pass {n+1}. Skipping.")
            # Append empty arrays to maintain structure if a pass is skipped
            data['X'].append(np.array([]))
            data['Y'].append(np.array([]))
            data['U'].append(np.array([]))
            data['V'].append(np.array([]))
            continue # Move to the next pass

        # Initialize NumPy arrays
        X = np.zeros((N_rows, N_cols), dtype=float)
        Y = np.zeros((N_rows, N_cols), dtype=float)
        U = np.zeros((N_rows, N_cols), dtype=float)
        V = np.zeros((N_rows, N_cols), dtype=float)

        # Read data into arrays
        for i in range(N_rows):
            for j in range(N_cols):
                line_to_read_idx = data_start_line_idx + i * N_cols + j
                if line_to_read_idx >= len(lines):
                    print(f"Error: Not enough data lines for pass {n+1} at (row {i+1}, col {j+1}). Data truncated.")
                    # Handle incomplete data by breaking or filling with zeros
                    # Here we break and append what we have so far
                    X = X[:i, :j] if j > 0 else X[:i, :] # Adjust array size if needed
                    Y = Y[:i, :j] if j > 0 else Y[:i, :]
                    U = U[:i, :j] if j > 0 else U[:i, :]
                    V = V[:i, :j] if j > 0 else V[:i, :]
                    break

                line_content = lines[line_to_read_idx].strip()
                try:
                    elems = line_content.split(",")
                    if len(elems) != 4:
                        raise ValueError(f"Expected 4 elements, got {len(elems)}")
                    X[i, j] = float(elems[0])
                    Y[i, j] = float(elems[1])
                    U[i, j] = float(elems[2])
                    V[i, j] = float(elems[3])
                except ValueError as ve:
                    print(f"Warning: Could not parse data in line {line_to_read_idx + 1} ('{line_content}'). Error: {ve}. Setting to 0.0.")
                    # Values will remain 0.0 due to np.zeros initialization

            else: # This 'else' belongs to the inner 'for j' loop and executes if it completes without a break
                continue
            break # This 'break' belongs to the outer 'for i' loop if the inner loop broke

        data['X'].append(X)
        data['Y'].append(Y)
        data['U'].append(U)
        data['V'].append(V)

    return data



out=unpack_data("vec_000.dat")
step=10;
plt.figure(figsize=(6, 6))
plt.quiver(out['X'][0], out['Y'][0], out['U'][0], out['V'][0],units='dots',       # Arrow dimensions in dots (pixels)
                scale_units='dots', # Arrow length scaling in dots (pixels)
                scale=0.9,            # A reference scale (e.g., 1 data unit = 1 pixel)
                width=2,            # Shaft width in 'dots' (e.g., 2 pixels wide)
                headwidth=2,        # Head width as multiple of shaft width
                headlength=2,       # Head length as multiple of shaft width
                headaxislength=2)   # Head length at shaft intersection)
#plt.grid(True)
plt.axis('equal')
plt.show()


