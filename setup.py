"""
1/22/26
Setup functions
"""

def create_config() -> bool:
    """
    Create an empty config file with default labels
    :return: ``True`` if file is written, ``False`` otherwise (e.g. if config file already exists)
    :rtype: bool
    """
    config = ("log_file:\n"
              "database:\n"
              "mni_template:\n"
              "raw:\n\n"
              "padded_image:\n"
              "mni_registered_image:\n"
              "bias_corrected_image:\n"
              "preprocessed_image:\n\n"
              "padded_label:\n"
              "mni_registered_label:\n"
              "imputed_label:\n"
              "preprocessed_label:\n\n"
              "unet:\n"
              "  train:\n"
              "  validation:\n"
              "  train_config:\n"
              "  test_config:\n")

    try:
        with open("config.yaml", "x") as f:
            f.write(config)
        return True
    except FileExistsError:
        pass

    return False

if __name__ == "__main__":
    create_config()