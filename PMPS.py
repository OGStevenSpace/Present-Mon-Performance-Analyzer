import asyncio
from testtest import request
import Main
import pandas as pd
import json


def load_json(file_path) -> (dict, str):
    """Load data from the JSON file"""
    try:
        with open(file_path) as file:
            return json.load(file), None
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        return None, str(e)


async def process_row(stats, system_prompts):
    return {
        key: await request(
            str(stats[key]),
            f"Describe the provided data. Reasoning: {system_prompts[key]}",
            deep_think=False
        )
        for key in system_prompts.keys()
    }


async def main_async():
    file_paths, num_files = Main.open_files()
    output_dict = {}
    system_prompts, status = load_json("perfo_prompts.json")

    tasks = []
    for file_path in file_paths:
        data, status = load_json(file_path)
        df = pd.DataFrame.from_dict(data)
        for i, val in enumerate(df.iterrows()):
            pc, stats = val
            tasks.append(process_row(stats, system_prompts))

    results = await asyncio.gather(*tasks)

    # Combine results into output_dict
    for idx, pc in enumerate(output_dict.keys()):
        output_dict[pc] = results[idx]

    return output_dict


if __name__ == "__main__":
    print(asyncio.run(main_async()))
