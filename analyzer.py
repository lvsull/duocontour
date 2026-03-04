import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def log_to_csv(log_file: str, train_output_path: str, val_output_path: str):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    lines = [re.sub(r".*UNetTrainer - ", "", line) for line in lines if re.search(r".*UNetTrainer.*Loss.*Evaluation score.*", line)]

    def create_csv(lines, keyword):
        df = pd.DataFrame(columns=["Iterations", "Loss", "Score"])
        filtered_lines = [line for line in lines if re.search(keyword, line)]
        iter = 0
        for line in filtered_lines:
            iter += 5000 if keyword == "Training" else 25000
            search = re.search(r".*Loss: (\d*.\d*).*Evaluation score: (\d*.\d*)", line)
            if search:
                df.loc[len(df)] = [iter, float(search.group(1)), float(search.group(2))]
        return df

    train_lines = create_csv(lines, "Training")
    val_lines = create_csv(lines, "Validation")

    train_lines.to_csv(train_output_path, index=False)
    val_lines.to_csv(val_output_path, index=False)


def graph_loss_score(csv_file: str, ax1, ax2, title: str = "", y_label: bool = False):
    data = pd.read_csv(csv_file)
    # data = data.iloc[:, ]
    sns.lineplot(data, x="Iterations", y="Loss", color="b", ax=ax1)
    sns.lineplot(data, x="Iterations", y="Score", color="r", ax=ax2)
    ax1.set_yscale("log")
    ax1.set_title(title)
    if not y_label:
        ax1.set_ylabel(None)
        ax2.set_yticklabels([])
        ax2.set_ylabel(None)
    ax2.set_ylim([0, 1])


if __name__ == "__main__":
anal    log_to_csv("info.log", "output/train.csv", "output/val.csv")
    sns.set_theme(style="whitegrid")
    sns.set_color_codes("bright")
    f, axs = plt.subplots(2, 2, sharex=True)
    f.set_dpi(600)
    f.set_size_inches(8, 8)
    graph_loss_score("output/train.csv", axs[0, 0], axs[1, 0], "Training", y_label=True)
    graph_loss_score("output/val.csv", axs[0, 1], axs[1, 1], "Validation")
    plt.suptitle("Loss and Evaluation Score vs Iteration for Training and Validation", size=16)
    f.tight_layout()
    plt.savefig("output/loss_score.png", dpi=600)
    plt.show()