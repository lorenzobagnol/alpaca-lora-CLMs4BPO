from typing import Literal
from tqdm import tqdm
import os
import pandas as pd


def load_product_dataset(dataset_path:str, cut_at:int)->pd.DataFrame:
    """
    Load the Amazon dataset containing information about products and their features.

    Args:
    """

    products=pd.read_csv(dataset_path)
    categories=products["category"].unique()
    # create a balanced dataset
    balanced_products=pd.DataFrame(columns=products.columns)
    for category in categories:
        balanced_products=balanced_products.append(products[products["category"]==category][:cut_at])
    return balanced_products

def generate_input(data:pd.DataFrame, return_type:Literal["once", "all"])->list:
    """
    Generate input for the Alpaca model to generate descriptions of luxury ship rooms containing products from the Amazon dataset.
    
    Args:
    cables: DataFrame containing information about cables.
    amps: DataFrame containing information about home audio products.
    teles: DataFrame containing information about televisions.
    return_type: string, can be "once" or "all".

    Returns:
    input: list of strings, each string is a sequence of products with their functionalities.
    """

    single_product_feature_list=list()
    for i in tqdm(range(len(data))):
        single_product_feature_list.append("<product> "+data["title"][i]+"\n"+"<features> "+data["feature"][i]+"\n\n")
    if return_type=="once":
        return single_product_feature_list
    