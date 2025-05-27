import PMPA
import PMPV
import datetime
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


def create_output_path(path):
    """Generate an output file path in the user's Documents directory."""
    timestamp = datetime.datetime.now().strftime('%d-%m-%y_T%H%M%S')
    return f"{path}/Output_{timestamp}.json"


def save_output(output_df, path):
    output_df.to_json(path)
    print(f"\nProcessing complete. Output file created at: {path}")


def main(out_flag=True):
    config = load_config()
    file_paths, num_files = open_files()
    output_path = create_output_path("/".join(file_paths[0].split("/")[:-1]))

    output_df = PMPA.main(file_paths, num_files, config)
    if out_flag:
        save_output(output_df, output_path)
    PMPV.main(output_df)


if __name__ == '__main__':
    main()
