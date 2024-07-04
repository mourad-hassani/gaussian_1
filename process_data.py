from pathlib import Path
import json
import csv

input_file_path = Path("./data/dates_dataset/date_dataset_close.json")
output_file_path = Path("./data/dates_dataset/dataset.csv")

with input_file_path.open("r") as f:
    input_data = json.load(f)

    with output_file_path.open("w") as g:
        writer = csv.writer(g, delimiter=",")
        writer.writerow(["sent0", "sent1", "score"])
        for line in input_data:
            line[0] = line[0].replace(";", "")
            line[1] = line[1].replace(";", "")
            if ";" in line[0]+line[1]:
                print(line)
            writer.writerow(line)