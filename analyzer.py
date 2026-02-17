import re
import pandas as pd

def log_to_csv(log_file: str, train_output_path: str, val_output_path: str):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    lines = [re.sub(r".*UNetTrainer - ", "", line) for line in lines if re.search(r".*UNetTrainer.*Loss.*Evaluation score.*", line)]

    def create_csv(lines, keyword):
        df = pd.DataFrame(columns=["Iterations", "Loss", "Score"])
        filtered_lines = [line for line in lines if re.search(keyword, line)]
        iter = 0
        for line in filtered_lines:
            iter += 5000 if keyword == "Training" else 10000
            search = re.search(r".*Loss: (\d*.\d*).*Evaluation score: (\d*.\d*)", line)
            if search:
                df.loc[len(df)] = [iter, float(search.group(1)), float(search.group(2))]
        return df

    train_lines = create_csv(lines, "Training")
    val_lines = create_csv(lines, "Validation")

    train_lines.to_csv(train_output_path, index=False)
    val_lines.to_csv(val_output_path, index=False)