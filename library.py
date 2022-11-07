import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import warnings
from sklearn.metrics import f1_score#, balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ----------------------------------------------------------------------------------------------------------------------------------------------------
#                               FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def find_random_state(features_df, labels, n=200):
  var = []  #collect test_error/train_error where error based on F1 score
  
  from sklearn.neighbors import KNeighborsClassifier
  model = KNeighborsClassifier(n_neighbors=5)

  #2 minutes
  for i in range(1, n):
      train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                      random_state=i, stratify=labels)
      model.fit(train_X, train_y)  #train model
      train_pred = model.predict(train_X)           #predict against training set
      test_pred = model.predict(test_X)             #predict against test set
      train_f1 = f1_score(train_y, train_pred)   #F1 on training predictions
      test_f1 = f1_score(test_y, test_pred)      #F1 on test predictions
      f1_ratio = test_f1/train_f1          #take the ratio
      var.append(f1_ratio)

  rs_value = sum(var)/len(var)  #get average ratio value
  idx = np.array(abs(var - rs_value)).argmin()  #find the index of the smallest value
  return idx

# ----------------------------------------------------------------------------------------------------------------------------------------------------
#                               CLASSES: TRANSFORMERS
# ----------------------------------------------------------------------------------------------------------------------------------------------------

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
  
# ----------------------------------------------------------------------------------------------------------------------------------------------------

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

    X_ = X.copy()
    
    if self.action == "keep":
        # use sets instead of for loop
        assert set(self.column_list) - set(X.columns.to_list()) == set(), f'''{self.__class__.__name__}.transform unknown column(s) 
        "{set(self.column_list) - set(X.columns.to_list())}" in: "{self.column_list}"'''
        # invert columns on keep
        colsToDrop = list(set(X_.columns.to_list()) - set(self.column_list))

    elif self.action == "drop":
        if all([self.column_list in X.columns.to_list()]) == False:
            warnings.warn(f'{self.__class__.__name__} Warning: one or more columns in column_list not present in dataframe.')
        colsToDrop = self.column_list

    else:
      print("Drop/Keep Control Flow error")
    
    X_ = X_.drop(columns = colsToDrop, inplace=False, errors="ignore")
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
# ----------------------------------------------------------------------------------------------------------------------------------------------------

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
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}" in {X.columns.to_list()}'  #column legit?

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
  
# ----------------------------------------------------------------------------------------------------------------------------------------------------

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
  
# ----------------------------------------------------------------------------------------------------------------------------------------------------

class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_name, numSigma = 3):  
    assert numSigma >=0, f'{self.__class__.__name__} sigma amount must be nonnegative but got {numSigma} instead.'
    assert isinstance(column_name, str), f'{self.__class__.__name__} column_name must be of type str but got {type(column_name)} instead.'
    self.numSigma = numSigma
    self.column_name = column_name

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.column_name in X.columns.to_list(), f'unknown column {self.column_name} in {X.columns}'
    assert all([isinstance(v, (int, float)) for v in X[self.column_name].to_list()])

    X_ = X.copy()
    sig = X_[self.column_name].std()
    mu = X_[self.column_name].mean()
    bounds = mu - self.numSigma * sig, mu + self.numSigma * sig
    X_[self.column_name] = X_[self.column_name].clip(lower=bounds[0], upper=bounds[1])
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

# ----------------------------------------------------------------------------------------------------------------------------------------------------

class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence = "outer"):  
    assert isinstance(target_column, str), f'{self.__class__.__name__} target_column must be of type str but got {type(target_column)} instead.'
    assert isinstance(fence, str), f'{self.__class__.__name__} tukey method specifier must be of type str but got {type(fence)} instead.'
    self.target_column = target_column
    self.fence = fence

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column} in {X.columns}'

    X_ = X.copy()
    q1 = X_[self.target_column].quantile(0.25)
    q3 = X_[self.target_column].quantile(0.75)  
    iqr = q3-q1
    inner_low = q1-1.5*iqr
    inner_high = q3+1.5*iqr
    outer_low = q1-3*iqr
    outer_high = q3+3*iqr
    
    if self.fence == "inner":
      X_[self.target_column] = X_[self.target_column].clip(inner_low, inner_high)
    elif self.fence == "outer":
      X_[self.target_column] = X_[self.target_column].clip(outer_low, outer_high)
    else:
      print("clip error")

    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

# ----------------------------------------------------------------------------------------------------------------------------------------------------

class MinMaxTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass  #takes no arguments

  #fill in rest below
  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    
    X_ = X.copy()
    cols = X_.columns.tolist() # save column names
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    numpy_result = scaler.fit_transform(X_)  #does not return a dataframe!
    X_ = pd.DataFrame(numpy_result, columns = cols) # add column names back in
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

# ----------------------------------------------------------------------------------------------------------------------------------------------------

class KNNTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, n_neighbors=5, weights="uniform"):
    assert isinstance(n_neighbors, int), f'KNNTransformer Error: n_neighbors must be of type int, got {type(n_neighbors)} instead.'
    assert isinstance(n_neighbors, int), f'KNNTransformer Error: weights must be of type str, got {type(weights)} instead.'
    self.n_neighbors = n_neighbors
    self.weights = weights

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    
    X_ = X.copy()
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights, add_indicator=False) 
    imputed_data = imputer.fit_transform(X_)
    X_ = pd.DataFrame(imputed_data, columns = X_.columns)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

# ----------------------------------------------------------------------------------------------------------------------------------------------------
#                               DATASET-SPECIFIC TRANSFORMERS AND SETUP FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------------------------------------

titanic_transformer = Pipeline(steps=[
    ('drop', DropColumnsTransformer(['Age', 'Gender', 'Class', 'Joined', 'Married',  'Fare'], 'keep')),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', MappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe', OHETransformer(target_column='Joined')),
    ('age', TukeyTransformer(target_column='Age', fence='outer')), #from chapter 4
    ('fare', TukeyTransformer(target_column='Fare', fence='outer')), #from chapter 4
    ('minmax', MinMaxTransformer()),  #from chapter 5
    ('imputer', KNNTransformer())  #from chapter 6
    ], verbose=True)

# ----------------------------------------------------------------------------------------------------------------------------------------------------

customer_transformer = Pipeline(steps=[
    ('id', DropColumnsTransformer(column_list=['ID'])),  #you may need to add an action if you have no default
    ('os', OHETransformer(target_column='OS')),
    ('isp', OHETransformer(target_column='ISP')),
    ('level', MappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('time spent', TukeyTransformer('Time Spent', 'inner')),
    ('minmax', MinMaxTransformer()),
    ('imputer', KNNTransformer())
    ], verbose=True)

# ----------------------------------------------------------------------------------------------------------------------------------------------------

def dataset_setup(full_table, label_column_name:str, the_transformer, rs, ts=.2):
  features = full_table.drop(columns=label_column_name)
  labels = full_table[label_column_name].to_list()

  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=ts, shuffle=True,
                                                    random_state=rs, stratify=labels)
  
  X_train_transformed = the_transformer.fit_transform(X_train)
  X_test_transformed = the_transformer.fit_transform(X_test)

  import numpy as np
  x_trained_numpy = X_train_transformed.to_numpy()
  x_test_numpy = X_test_transformed.to_numpy()
  y_train_numpy = np.array(y_train)
  y_test_numpy = np.array(y_test)

  return x_trained_numpy, x_test_numpy, y_train_numpy, y_test_numpy

# ----------------------------------------------------------------------------------------------------------------------------------------------------

def titanic_setup(titanic_table, transformer=titanic_transformer, rs=40, ts=.2):
  x_trained_numpy, x_test_numpy, y_train_numpy, y_test_numpy = dataset_setup(titanic_table, 'Survived',
                                                                           transformer, rs=rs, ts=ts)
  return x_trained_numpy, x_test_numpy, y_train_numpy, y_test_numpy

# ----------------------------------------------------------------------------------------------------------------------------------------------------

def customer_setup(customer_table, transformer=customer_transformer, rs=76, ts=.2):
  x_trained_numpy, x_test_numpy, y_train_numpy, y_test_numpy = dataset_setup(customer_table, 'Rating',
                                                                           transformer, rs=rs, ts=ts)
  return x_trained_numpy, x_test_numpy, y_train_numpy, y_test_numpy
