import pandas as pd
from pathlib import Path

print("Cleaning ARFF safely...")

file_path = Path("Chronic_Kidney_Disease/chronic_kidney_disease.arff")

if not file_path.exists():
    print(f"Error: Dataset not found at {file_path}")
    exit(1)

data_rows = []
columns = []
data_section = False

with open(file_path, "r") as f:
    for line in f:
        line = line.strip()
        
        if line.lower().startswith("@attribute"):
            parts = line.split()
            
            col_name = parts[1].strip("'")
            columns.append(col_name)
        
        if line.lower().startswith("@data"):
            data_section = True
            continue
       
        if data_section and line != "":
            row = [x.strip() for x in line.split(",")]
           
            if len(row) > len(columns):
                row = row[:len(columns)]

            data_rows.append(row)

df = pd.DataFrame(data_rows, columns=columns)


df = df.replace("?", pd.NA)

output_path = Path("ckd.csv")
df.to_csv(output_path, index=False)

print(f"Successfully converted to {output_path}")
print(f"Final Shape: {df.shape}")