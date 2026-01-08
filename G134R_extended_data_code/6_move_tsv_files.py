import os
import shutil


data_folder = r"Z:/personal_data/Filip_Liebsch/SP8_Biocenter_small/2025-04-23_floxed/R670A_D456A_project/data/"

destination_folder = r"Z:/personal_data/Filip_Liebsch/SP8_Biocenter_small/2025-04-23_floxed/R670A_D456A_project/analysis_summary/data/"

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Walk through all directories and subdirectories
for root, dirs, files in os.walk(data_folder):
    # Check if current directory is named "measurements"
    if os.path.basename(root) == "measurements":
        # Look for .tsv files in this measurements folder
        for file in files:
            if file.endswith(".tsv") and "cluster_summary" in file.lower():
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder, file)
                
                # Copy the file
                shutil.copy2(source_path, destination_path)
                print(f"Copied: {file} from {root}")

print(f"\nAll .tsv files from 'measurements' folders have been copied to {destination_folder}")
