import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import warnings

# one hot encoding abbv: OHE
class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):  
    assert isinstance(target_column, str), f'{self.__class__.__name__} constructor expected String but got {type(target_column)} instead.'
    self.target_column = target_column 
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  #column legit?
    
    X_ = X.copy()
    X_ = pd.get_dummies(X,
                        prefix=self.target_column.replace(" ", "-"), #replace whitespace 
                        prefix_sep='_',     
                        columns=[self.target_column],
                        dummy_na=self.dummy_na,    
                        drop_first=self.drop_first    
                        )
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'{self.__class__.__name__} action {action} not in ["keep", "drop"]'
    assert all([isinstance(col, str) for col in column_list]), "All entries in column_list must be of type str."
    self.column_list = column_list
    self.action = action

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    if self.action == "keep":
      # use sets instead of for loop
      assert set(self.column_list) - set(X.columns.to_list()) == set(), f'{self.__class__.__name__}.transform unknown column(s) "{set(self.column_list) - set(X.columns.to_list())}" in: "{self.column_list}"'
      # invert columns on keep
      self.column_list = list(set(X.columns.to_list()) - set(self.column_list))

    if self.action == "drop":
      if all([self.column_list in X.columns.to_list()]) == False:
        warnings.warn(f'{self.__class__.__name__} Warning: one or more columns in column_list not present in dataframe.')
    
    X_ = X.copy()
    X_.drop(columns = self.column_list, inplace=True, errors="ignore")     
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class MappingTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?

    #now check to see if all keys are contained in column
    column_set = set(X[self.mapping_column])
    keys_not_found = set(self.mapping_dict.keys()) - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  set(self.mapping_dict.keys())
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class PearsonTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, threshold = 0.4):  
    assert 0 <= threshold <= 1, f'{self.__class__.__name__} threshold must be in [0,1] but got {threshold} instead.'
    self.threshold = threshold

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    transformed_df = X.copy()
    # correlation matrix - always square
    df_corr = transformed_df.corr(method='pearson')
    dim = df_corr.shape[0]
    # True if abs value of corr is > threshold
    threshBool = (abs(df_corr) >= self.threshold)
    # get upper triangluar mask of booleans (k=1 to also drop the diagonal)
    maskTemplate = np.arange(1*(dim**2)).reshape(dim,dim) >= 0
    upTriMask = np.triu(maskTemplate, k=1)
    # extract upper triangle of correlations over threshold and extract correlated columns
    threshBoolUpTri = upTriMask & threshBool 
    correlated_columns = [col for _, col in enumerate(threshBoolUpTri) if any(threshBoolUpTri[col]) == True] 
    # drop those columns and return
    new_df = transformed_df.drop(columns=correlated_columns)
    return new_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
