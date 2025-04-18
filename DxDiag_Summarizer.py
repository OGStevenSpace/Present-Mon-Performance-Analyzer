import re
import csv
import json
import sys
import tkinter as tk
from tkinter import filedialog


def open_files():
    """Open file dialog for selecting DxDiag text files."""
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select DxDiag Text Files",
                                             filetypes=[("Text Files", "*.txt")])
    return file_paths, len(file_paths)


def extract_section(text, header):
    """
    Extract the block of text under the given header (e.g. "System Information").
    It assumes that the section header is surrounded by dashed lines.
    """
    pattern = re.compile(r"-+\s*\n\s*" + re.escape(header) + r"\s*\n-+\s*\n(.*?)(?=\n-{3,}\n|$)", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1)
    return ""


def extract_field(block, field_name):
    """
    Given a text block and a field name, extract the value (text after the colon).
    """
    pattern = re.compile(r"^ *" + re.escape(field_name) + r"\s*:\s*(.*)$", re.MULTILINE)
    match = pattern.search(block)
    if match:
        return match.group(1).strip()
    return ""


def extract_system_information(text):
    sys_info_block = extract_section(text, "System Information")
    fields = [
        "Time of this report",
        "Machine name",
        "Operating System",
        "Language",
        "System Manufacturer",
        "System Model",
        "Processor",
        "Memory",
        "Page File",
        "DirectX Version",
        "Mux Target GPU"
    ]
    data = {}
    for field in fields:
        data[field] = extract_field(sys_info_block, field)
    return data


def extract_display_devices(text):
    disp_section = extract_section(text, "Display Devices")
    # Updated: split the section by blank lines
    devices = [block for block in disp_section.split("\n\n") if "Card name" in block]
    result = []
    fields = [
        "Card name",
        "Chip type",
        "Display Memory",
        "Dedicated Memory",
        "Shared Memory",
        "Current Mode",
        "HDR Support",
        "Native Mode",
        "Driver Version",
        "Driver Date/Size"
    ]
    for device in devices:
        entry = {}
        for field in fields:
            entry[field] = extract_field(device, field)
        # Only add if at least one field has non-empty data
        if any(entry.values()):
            result.append(entry)
    return result


def extract_disk_drives(text):
    disk_section = extract_section(text, "Disk & DVD/CD-ROM Drives")
    # Split by looking for lines that start with "Drive:" (using MULTILINE mode)
    drive_blocks = re.split(r"(?=^\s*Drive\s*:)", disk_section, flags=re.MULTILINE)
    result = []
    fields = [
        "Drive",
        "Free Space",
        "Total Space",
        "File System",
        "Model"
    ]
    for block in drive_blocks:
        block = block.strip()
        if not block:
            continue
        entry = {}
        for field in fields:
            entry[field] = extract_field(block, field)
        if entry.get("Drive"):
            result.append(entry)
    return result


def process_file(filepath):
    """Extract fields from a single file and return a dict for CSV output."""
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    sys_info = extract_system_information(text)
    display_devices = extract_display_devices(text)
    disk_drives = extract_disk_drives(text)
    sys_info["Source File"] = filepath
    sys_info["Display Devices"] = json.dumps(display_devices, ensure_ascii=False)
    sys_info["Disk Drives"] = json.dumps(disk_drives, ensure_ascii=False)
    return sys_info


def main():
    file_paths, count = open_files()
    if count == 0:
        print("No files selected. Exiting.")
        return

    root = tk.Tk()
    root.withdraw()
    output_csv = filedialog.asksaveasfilename(title="Save CSV File", defaultextension=".csv",
                                              filetypes=[("CSV Files", "*.csv")])
    if not output_csv:
        print("No output file selected. Exiting.")
        return

    base_fields = [
        "Source File",
        "Time of this report",
        "Machine name",
        "Operating System",
        "Language",
        "System Manufacturer",
        "System Model",
        "Processor",
        "Memory",
        "Page File",
        "DirectX Version",
        "Mux Target GPU",
        "Display Devices",
        "Disk Drives"
    ]

    rows = []
    for filepath in file_paths:
        row = process_file(filepath)
        rows.append(row)

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=base_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in base_fields})
    print(f"Data extracted from {count} file(s) to {output_csv}")


if __name__ == '__main__':
    main()
