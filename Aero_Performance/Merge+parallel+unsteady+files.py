# merge files
import pandas as pd
import os
import csv

sim_data_path = "D:\\com_files\\KARI_Airfoil\\sol_Test\\"
merged_data_path = "D:\\com_files\\KARI_Airfoil\\sol_Test\\"
Tecplot_header = "variables=X, Y, Z, Vor, Q_cri"

# create directory if not exist
os.makedirs(os.path.dirname(merged_data_path), exist_ok=True)

# list of file names
for t in range(1,2):
    filenames = []
    merged = []
    for i in range(1,9):
        filenames.append(sim_data_path+"flo0"+str(i).rjust(2,'0')+".00000"+str(t).rjust(2, '0')+"uns") 
    print(filenames)
    allData = []
    for f in filenames:
        df = pd.read_csv(f, na_filter=True, skipinitialspace=True) 
        allData.append(df)
    print(df.shape)
    combined_csv = pd.concat( allData, axis=0, ignore_index=True )
    print(combined_csv.shape)
    print(combined_csv.tail())
    Tecplot_header.split(',', maxsplit=combined_csv.shape[1]-1)
    csv = combined_csv.to_csv(merged_data_path+"combined_flo.00000"+str(t).rjust(2,'0')+"uns", header=Tecplot_header, sep=' ', index=False, quotechar=' ' )

print('"x","y"')