def splitFeatsTraget(df: "pd.dataFrame") -> "pd.dataFrame":
    """Split the data into training and test set
    
    Args: df (pd.DataFrame): raw data
        
    Process:
        1. Set the feature names
        2. Split the data into training and test
    
    Return: ["pd.dataFrame", "pd.dataFrame"]
    """
    
    # 1. Set the feature names
    FEATS = [
        'P1_MAR_STATUS_P', 'PAYMENT_METHOD_NonDD', 'HOME_EM_ADDON_POST_REN_Y',
       'AD_BUILDINGS_Y', 'KEYCARE_ADDON_PRE_REN_Y', 'MAX_DAYS_UNOCC',
       'HP1_ADDON_POST_REN_Y', 'HP2_ADDON_POST_REN_Y',
       'LEGAL_ADDON_POST_REN_Y', 'PAYMENT_METHOD_PureDD',
       'HP3_ADDON_POST_REN_Y', 'SUM_INSURED_BUILDINGS',
       'GARDEN_ADDON_PRE_REN_Y', 'KEYCARE_ADDON_POST_REN_Y', 'OCC_STATUS_LP',
       'GARDEN_ADDON_POST_REN_Y', 'HOME_EM_ADDON_PRE_REN_Y', 'MTA_FLAG_Y',
       'PROP_TYPE_2.0', 'P1_MAR_STATUS_O', 'PROP_TYPE_22.0',
       'NCD_GRANTED_YEARS_B', 'SUBSIDENCE_Y', 'cover_length', 'OCC_STATUS_PH',
       'OWNERSHIP_TYPE_2.0', 'PROP_TYPE_51.0', 'SUM_INSURED_CONTENTS',
       'BUILDINGS_COVER_Y', 'RISK_RATED_AREA_C_imputed',
       'RISK_RATED_AREA_B_imputed', 'NCD_GRANTED_YEARS_C', 'SAFE_INSTALLED_Y',
       'MTA_FAP_imputed', 'PROP_TYPE_16.0', 'UNSPEC_HRP_PREM', 'PROP_TYPE_4.0',
       'OCC_STATUS_UN', 'P1_EMP_STATUS_H', 'FLOODING_Y', 'PROP_TYPE_17.0',
       'LAST_ANN_PREM_GROSS', 'SPEC_ITEM_PREM', 'BUS_USE_Y', 'PROP_TYPE_18.0',
       'P1_MAR_STATUS_D', 'PROP_TYPE_32.0', 'MTA_APRP_imputed',
       'HP2_ADDON_PRE_REN_Y', 'SPEC_SUM_INSURED', 'OWNERSHIP_TYPE_7.0',
       'CLAIM3YEARS_Y', 'PROP_TYPE_45.0', 'PROP_TYPE_53.0', 'P1_EMP_STATUS_U',
       'P1_MAR_STATUS_M', 'P1_EMP_STATUS_S', 'property_age', 'P1_MAR_STATUS_W',
       'PROP_TYPE_26.0', 'PROP_TYPE_7.0', 'P1_MAR_STATUS_S', 'P1_EMP_STATUS_N',
       'PROP_TYPE_9.0', 'PROP_TYPE_52.0', 'age', 'P1_MAR_STATUS_C',
       'OWNERSHIP_TYPE_14.0', 'BEDROOMS', 'PROP_TYPE_25.0', 'PROP_TYPE_19.0',
       'OWNERSHIP_TYPE_18.0', 'P1_EMP_STATUS_E', 'APPR_ALARM_Y',
       'OWNERSHIP_TYPE_12.0', 'OWNERSHIP_TYPE_3.0', 'PROP_TYPE_48.0',
       'APPR_LOCKS_Y', 'P1_EMP_STATUS_R', 'SEC_DISC_REQ_Y',
       'OWNERSHIP_TYPE_13.0', 'OWNERSHIP_TYPE_8.0', 'P1_SEX_M',
       'PROP_TYPE_10.0', 'PROP_TYPE_47.0', 'NEIGH_WATCH_Y'
        ]
    
    # 2. Split the data into training and test
    X, y = df[FEATS], df["lapse"]
    
    return [X, y]