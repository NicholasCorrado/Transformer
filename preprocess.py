import numpy as np

def prune_data(file_path, out_file_path):

    data = np.genfromtxt(fname=file_path,
                            delimiter=',',
                            skip_header=True,
                            filling_values=-1)
    data = data.T

    # Read first line to get column names
    file = open(file_path)
    col_names = file.readline().split(',')
    data_out = []
    col_indices = []


    for i in range(len(data)):
        col = data[i]
        no_data = col == -1
        num_no_data = np.sum(no_data)
        if num_no_data == 0:
            col_indices.append(i)

    col_indices = np.array(col_indices, dtype=np.int64)
    col_names = np.array(col_names)
    col_names_no_null = col_names[col_indices]

    constituents = np.genfromtxt(
        fname="data/s&p_constituents.csv",
        delimiter=',',
        skip_header=True,
        dtype=str,
        usecols=(0,))

    col_indices_no_null_sp500 = []
    for i, name in enumerate(col_names_no_null):
        if name in constituents:
            col_indices_no_null_sp500.append(col_indices[i])

    col_indices_no_null_sp500 = np.array(col_indices_no_null_sp500)
    col_names_no_nul_sp500 = np.array(col_names[col_indices_no_null_sp500])
    print(col_names_no_nul_sp500)

    data_no_null_sp500 = data[col_indices_no_null_sp500]
    np.save(out_file_path, data_no_null_sp500)
    np.save(f'{out_file_path}_names', col_names_no_nul_sp500)


if __name__ == "__main__":
    prune_data('./data/energydata.csv', './data/energy_pruned')
    prune_data('./data/techdata.csv', './data/tech_pruned')



