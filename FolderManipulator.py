import os
import shutil
import tkinter as tk
from tkinter import filedialog

def move_files_to_target(source_dir, target_dir):
    for root, dirs, files in os.walk(source_dir, topdown=False):
        # Construct the folder path prefix
        rel_path = os.path.relpath(root, source_dir)
        if rel_path == ".":
            folder_prefix = ""
        else:
            folder_prefix = rel_path.replace(os.sep, "_") + "_"

        # Move files to the target directory with the prefixed name
        for file in files:
            src_file_path = os.path.join(root, file)
            new_file_name = folder_prefix + file
            dest_file_path = os.path.join(target_dir, new_file_name)
            shutil.move(src_file_path, dest_file_path)
            print(f"Moved: {src_file_path} to {dest_file_path}")

        # Remove the directory if it is empty
        if not os.listdir(root):
            os.rmdir(root)
            print(f"Removed empty directory: {root}")

if __name__ == "__main__":

    root = tk.Tk()
    root.withdraw()
    source_directory = filedialog.askdirectory()
    destination_directory = filedialog.askdirectory()

    # Ensure target directory exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    move_files_to_target(source_directory, destination_directory)
