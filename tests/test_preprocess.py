import pandas as pd

from src.modules.transform.preprocess import prepareInputs

def test_preprocess():
    """Test proprocess function
    
        Input: a whole row dat
        
        Expected: shape = (189005, 177)
    """
    
    df = pd.read_csv("resources/data/home_insurance.csv")
    
    df = prepareInputs(df)
    
    assert df.shape == (189005, 177)