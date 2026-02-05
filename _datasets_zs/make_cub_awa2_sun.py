import pickle

import numpy as np
import scipy.io as scio


def print_all(mat_data):
    for x in mat_data:
        if isinstance(mat_data[x], bytes) or isinstance(mat_data[x], str):
            print(x, mat_data[x])
        elif isinstance(mat_data[x], list):
            print(x, len(mat_data[x]))
        else:
            print(x, mat_data[x].shape)


def save_concepts(database, mat_data):
    data = mat_data["allclasses_names"]
    with open(
        f"C:/Users/QQ/OneDrive/DEV/Python/pytorch/HASH-ZOO/_datasets_zs/{database}/concepts.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for item in data:
            f.write(str(item[0][0]) + "\n")


def save_img_list(database, mat_data):
    data = mat_data["image_files"]

    if database == "awa2":
        rpl_str = "/BS/xian/work/data/Animals_with_Attributes2//JPEGImages/"
    elif database == "cub":
        rpl_str = "/BS/Deep_Fragments/work/MSc/CUB_200_2011/CUB_200_2011/images/"
    elif database == "sun":
        rpl_str = "/BS/Deep_Fragments/work/MSc/data/SUN/images/"
    else:
        raise NotImplementedError

    with open(
        f"C:/Users/QQ/OneDrive/DEV/Python/pytorch/HASH-ZOO/_datasets_zs/{database}/img_list.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for item in data:
            f.write(item[0][0].replace(rpl_str, "") + "\n")


def save_lab_list(database, mat_data):
    data = mat_data["labels"]
    with open(
        f"C:/Users/QQ/OneDrive/DEV/Python/pytorch/HASH-ZOO/_datasets_zs/{database}/lab_list.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for item in data:
            f.write(str(item[0]) + "\n")


def save_att_list(database, mat_data):
    data = mat_data["original_att"]
    np.save(
        f"C:/Users/QQ/OneDrive/DEV/Python/pytorch/HASH-ZOO/_datasets_zs/{database}/att_list.npy",
        data.astype(np.float32).T,
    )


def save_idx_list(database, mat_data):
    trainval_loc = mat_data["trainval_loc"].squeeze() - 1
    test_seen_loc = mat_data["test_seen_loc"].squeeze() - 1
    test_unseen_loc = mat_data["test_unseen_loc"].squeeze() - 1

    with open(f"C:/Users/QQ/OneDrive/DEV/Python/pytorch/HASH-ZOO/_datasets_zs/{database}/idx_list.pkl", "wb") as f:
        pickle.dump(
            {"trainval_loc": trainval_loc, "test_seen_loc": test_seen_loc, "test_unseen_loc": test_unseen_loc}, f
        )


if __name__ == "__main__":

    base_dir = "D:/Test/Datasets/xlsa17/data"

    for database in ["awa2", "cub", "sun"]:

        mat_data = scio.loadmat(f"{base_dir}/{database.upper()}/att_splits.mat")
        # print_all(mat_data)
        # break
        # print(mat_data["train_loc"].dtype)
        # save_idx_list(database, mat_data)
        # save_concepts(database, mat_data)
        save_att_list(database, mat_data)

        # mat_data = scio.loadmat(f"{base_dir}/{database.upper()}/res101.mat")
        # save_img_list(database, mat_data)
        # save_lab_list(database, mat_data)
        # print_all(mat_data)
        # print(mat_data["labels"][0][0])
        # break
