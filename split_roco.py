import pandas as pd


def get_test_data(path):
    """
    :param path: path to the file that include ROCO's all_data
    :return: test dataset as list
    """
    df = pd.read_csv(path+"\\test\\radiologytestdata.csv").drop(columns=["id"])
    df["name"]=path+"\\test\\radiology\\images\\"+df["name"]
    return df.values.tolist()


def get_train_data(path):
    """
    :param path: path to the file that include ROCO's all_data
    :return: train dataset as list
    """
    df = pd.read_csv(path+"\\train\\radiologytraindata.csv").drop(columns=["id"])
    df["name"]=path+"\\train\\radiology\\images\\"+df["name"]
    return df.values.tolist()


def get_validation_data(path):
    """
    :param path: path to the file that include ROCO's all_data
    :return: validation dataset as list
    """
    df = pd.read_csv(path+"\\validation\\radiologyvaldata.csv").drop(columns=["id"])
    df["name"] = path+"\\validation\\radiology\\images\\" + df["name"]
    return df.values.tolist()
