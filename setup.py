"""
1/22/26
Setup functions
"""

import yaml

def add_to_line(initial: str, new: str) -> str:
    return initial.strip("\n") + " " + new + "\n"

def fill_config():
    with open("config.yaml", "r") as f:
        lines = f.readlines()

    print("INPUT EACH DIRECTORY OR FILE PATH")

    for i in range(len(lines)):
        if lines[i] == "\n":
            print()
            continue
        try:
            if lines[i+1].startswith("  ") and not lines[i].startswith("  "):
                continue
        except IndexError:
            pass

        value = input(lines[i].strip() + " ")
        lines[i] = add_to_line(lines[i], value)

    with open("config.yaml", "w") as f:
        f.writelines(lines)


def fill_train_config():
    with open("config.yaml", "r") as f:
        unet_conf = yaml.safe_load(f)["unet"]
        config_path = unet_conf["train_config"]
        checkpoint_path = unet_conf["checkpoint"]
        train_path = unet_conf["train"]
        val_path = unet_conf["validation"]

    with open(config_path, "r") as f:
        lines = f.readlines()

    lines[23] = add_to_line(lines[23], checkpoint_path)
    lines[86] = add_to_line(lines[86], train_path)
    lines[136] = add_to_line(lines[136], val_path)

    with open(config_path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    fill_config()
    fill_train_config()