# Imports
import numpy as np
import pandas as pd
import utils.NERutils as NU 


#****************************
#***   Test script   ***
#****************************

# evaluate predictions on gold labels
def testPreds(predPath: str, labelsPath: str) -> float:
    """
    Reads the predictions and gold labels from a path

    :param predPath: path to file containing predictions. Must be .parquet format
    :param labelsPath: path to file containing gold labels. Must be .parquet format

    :returns: span F1-score
    """

    preds = NU.readDataset(predPath)
    test = NU.readDataset(labelsPath)

    predsEntList = NU.getEntsForPredictions(test)






if __name__ == '__main__':
    main()