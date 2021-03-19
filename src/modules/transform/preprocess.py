import pandas as pd
import numpy as np
from datetime import datetime


def prepareInputs(df: "pd.dataFrame") -> "pd.dataFrame":
    """Prepare the input for training

    Args:
        df (pd.DataFrame): raw data
        
    Process:
        1. Exclude missing values
        2. Clean the target variable
        3. Create dummy variables for categorical variables
        4. Create age features
        5. Impute missing value
    
    Return: pd.dataFrame
    """
    
    # 1. Exclude missing values
    df = df[df["POL_STATUS"].notnull()]
    
    # 2. Clean the target variable
    df = df[df["POL_STATUS"] != "Unknown"]
    df["lapse"] = np.where(df["POL_STATUS"] == "Lapsed", 1, 0)
    
    # 3. Create dummy variables for categorical variables
    categorical_cols = ["CLAIM3YEARS", "BUS_USE", "AD_BUILDINGS",
                        "APPR_ALARM", "CONTENTS_COVER", "P1_SEX",
                        "BUILDINGS_COVER", "P1_POLICY_REFUSED", 
                        "APPR_ALARM", "APPR_LOCKS", "FLOODING",
                        "NEIGH_WATCH", "SAFE_INSTALLED", "SEC_DISC_REQ",
                        "SUBSIDENCE", "LEGAL_ADDON_POST_REN", 
                        "HOME_EM_ADDON_PRE_REN","HOME_EM_ADDON_POST_REN", 
                        "GARDEN_ADDON_PRE_REN", "GARDEN_ADDON_POST_REN", 
                        "KEYCARE_ADDON_PRE_REN", "KEYCARE_ADDON_POST_REN", 
                        "HP1_ADDON_PRE_REN", "HP1_ADDON_POST_REN",
                        "HP2_ADDON_PRE_REN", "HP2_ADDON_POST_REN", 
                        "HP3_ADDON_PRE_REN", "HP3_ADDON_POST_REN", 
                        "MTA_FLAG", 'OCC_STATUS', 'OWNERSHIP_TYPE',
                        'PROP_TYPE', 'PAYMENT_METHOD', "P1_EMP_STATUS",
                        "P1_MAR_STATUS"
                        ]
    
    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], 
                                 drop_first = True,
                                 prefix = col
                                )
        df = pd.concat([df, dummies], 1)
    
    # 4. Create age features
    df["age"] = (datetime.strptime("2013-01-01", "%Y-%m-%d") - pd.to_datetime(df["P1_DOB"])).dt.days // 365
    df["property_age"] = 2013 - df["YEARBUILT"]
    df["cover_length"] = 2013 - pd.to_datetime(df["COVER_START"]).dt.year
    
    # 5. Impute missing value
    df["RISK_RATED_AREA_B_imputed"] = df["RISK_RATED_AREA_B"].fillna(df["RISK_RATED_AREA_B"].mean())
    df["RISK_RATED_AREA_C_imputed"] = df["RISK_RATED_AREA_C"].fillna(df["RISK_RATED_AREA_C"].mean())

    return df
