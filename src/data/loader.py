import pandas as pd
import re
import string
import pandas as pd

def load_data(filepath):

  columns = ["target", "id", "date", "flag", "user", "text"]

  df = pd.read_csv(
      filepath,
      encoding="latin-1",
      names=columns
  )

  df = df[['text', 'target']]
  df["target"] = df["target"].replace(4,1)

  return df
