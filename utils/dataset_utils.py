from typing import Literal
from tqdm import tqdm
import os
import pandas as pd

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
    