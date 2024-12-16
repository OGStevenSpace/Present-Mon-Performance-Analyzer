import os
import csv
import tkinter as tk
from tkinter import filedialog


# Function to open folder dialog and select folder
def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(title="Select Folder")
    return folder_path


# Function to save filenames to CSV
def save_filenames_to_csv(folder_path):
    filenames = os.listdir(folder_path)
    filenames = [f for f in filenames if
                 os.path.isfile(os.path.join(folder_path, f)) and f != "file_names.csv"]  # Exclude file_names.csv

    # Save the filenames to a CSV file
    csv_file = os.path.join(folder_path, "file_names.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename"])  # Writing header
        for filename in filenames:
            writer.writerow([filename])


# Function to rename files, ensuring file_names.csv is not renamed
def rename_files(folder_path):
    filenames = os.listdir(folder_path)
    filenames = [f for f in filenames if
                 os.path.isfile(os.path.join(folder_path, f)) and f != "file_names.csv"]  # Exclude file_names.csv

    for index, filename in enumerate(filenames, start=1):
        new_name = f"Test {index:02d}{os.path.splitext(filename)[1]}"  # Generate the new filename with proper format
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_name)
        os.rename(old_file_path, new_file_path)


def main():
    # Step 1: Select folder
    folder_path = select_folder()
    if not folder_path:
        print("No folder selected. Exiting...")
        return

    # Step 2: Save filenames to CSV
    save_filenames_to_csv(folder_path)
    print(f"Filenames saved to {folder_path}/file_names.csv")

    # Step 3: Rename files
    rename_files(folder_path)
    print("Files renamed successfully.")


if __name__ == "__main__":
    main()