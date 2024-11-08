import PMPA
import PMPV
import datetime
import os
import json
import tkinter as tk
from tkinter import filedialog


def load_config():
    """Load configuration from the JSON file."""
    with open('config.json', "r") as file:
        return json.load(file)


def open_files():
    """Open file dialog for selecting CSV files."""
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames()
    return file_paths, len(file_paths)


def create_output_path():
    """Generate an output file path in the user's Documents directory."""
    timestamp = datetime.datetime.now().strftime('%d-%m-%y_T%H%M%S')
    return os.path.join(os.path.expanduser('~'), 'Documents', f"Output_{timestamp}.csv")


def save_output(output_df, path):
    """Save output DataFrame to specified CSV path."""
    file_exists = os.path.isfile(path)
    output_df.to_csv(path, mode='a', index=False, header=not file_exists)
    print(f"\nProcessing complete. Output file created at: {path}")

if __name__ == '__main__':

    config = load_config()
    output_path = create_output_path()
    file_paths, num_files = open_files()

    output_df = PMPA.main(file_paths, num_files, config)
    save_output(output_df, output_path)
    PMPV.main(output_df, config, sort=True)
