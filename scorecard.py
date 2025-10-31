import pandas as pd
import numpy as np
import json
import pickle

from optbinning import OptimalBinning
from IPython.display import clear_output
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

class ModelData(dict):
    def __getattr__(self, name):
        return self[name]
    
class Scorecard(object):
    def __init__(self, data, start_dateintime, end_dateintime, start_date_oot, end_date_oot, date_column='approvedDate', key='id', flag='flag'):
        self.data = data
        self.data_copy = data.copy()
        self.meta_data = {} 
        self.intime_binning = None 
        self.key = key
        self.flag = flag
        self.start_dateintime = start_dateintime
        self.end_dateintime = end_dateintime
        self.start_date_oot = start_date_oot
        self.end_date_oot = end_date_oot
        self.date_column = date_column
        self.intime_oot_split()
        self.default_woe = 0 
        cv = StratifiedShuffleSplit(n_splits=5,random_state=0)
        self.cv = cv
        self.PDO = 40
        self.ODDS = 50
        self.S0 = 500
        self._calculate_constants()

    def _calculate_constants(self):
        self.M = self.PDO / np.log(2.)
        self.C = self.S0 + (self.M * np.log(1./self.ODDS))

    def intime_oot_split(self):
        intime_df = self.data[(self.data[self.date_column] >= self.start_dateintime) & (self.data[self.date_column] <= self.end_dateintime)] 
        intime_df = intime_df.reset_index(drop=True)
        oot_df = self.data[(self.data[self.date_column] >= self.start_date_oot) & (self.data[self.date_column] <= self.end_date_oot)] 
        oot_df = oot_df.reset_index(drop=True)
        
        self.intime = ModelData(intime_df)
        self.oot = ModelData(oot_df)
        
        if self.intime_binning is None:
            self.intime_binning = intime_df[[self.key, self.flag]].copy()
            
    def intime_oot_split(self):
        intime_df = self.data[(self.data[self.date_column] >= self.start_dateintime) & (self.data[self.date_column] <= self.end_dateintime)].reset_index(drop=True)
        oot_df = self.data[(self.data[self.date_column] >= self.start_date_oot) & (self.data[self.date_column] <= self.end_date_oot)].reset_index(drop=True)

        self.intime = ModelData({'data': intime_df})
        self.oot = ModelData({'data': oot_df})

        self.intime[self.flag] = intime_df[self.flag]
        self.intime[self.key] = intime_df[self.key]
        self.oot[self.flag] = oot_df[self.flag]
        self.oot[self.key] = oot_df[self.key]

        if self.intime_binning is None:
            self.intime_binning = intime_df[[self.key, self.flag]].copy()
            
    def quantile_binning(series):
        binning_data = series.quantile(np.linspace(0, 0.9, 10)).unique().tolist()
        binning_data = [binning_data[0] - 1] + [-999,-99,-1,0] + binning_data + [np.inf]
        binning_data = sorted(set(binning_data))
        return binning_data
    
    def optimal_binning(self, col):
        data_type = type(self.intime.data[col].values[0])
        if data_type is np.float64 or dtype is np.int64:
            optbin = OptimalBinning(name=col, dtype="numerical", solver='cp')
            optbin.fit(self.intime.data[col].values, self.intime_binning.flag.values)
            splits = optbin.splits.tolist() if optbin.splits is not None else []
            splits = [-np.inf] + [-999,-99,-1,0] + splits + [np.inf]
            splits = sorted(set(splits))
            return splits
    
    def binning_builder(self,series, binning_method='quantile'):
        data_type = series.dtype
        categorical = (data_type != 'int64' and data_type != 'float64')
        meta_data = {}
        meta_data['default_null'] = -9999999999
        meta_data['data_type'] = "categorical" if categorical else "numerical" 
        series = series.fillna(meta_data['default_null'])
        binning_data = None
        if categorical:
            binning_data = {}
            for val in series.astype(str).unique():
                binning_data[val] = val
        else:
            if binning_method=='quantile':
                binning_data = Scorecard.quantile_binning(series)
            elif binning_method=='optimal':
                col_name = series.name
                binning_data = self.optimal_binning(col_name)
                
                
        meta_data['binning_data'] = binning_data
        return meta_data
    
    @staticmethod    
    def get_binning(raw_values, meta_data_):
        # handle null 
        raw_values = raw_values.fillna(meta_data_['default_null'])

        # binning mapping
        binning_data = meta_data_['binning_data']
        categorical = meta_data_['data_type'] == 'categorical'
        if categorical:
            bin_values = raw_values.astype(str).apply(lambda x: binning_data[x] if x in binning_data else np.nan)
        else:
            bin_values = pd.cut(raw_values, binning_data) 

        return bin_values

    @staticmethod
    def calculate_woe(data, index_column, target_column, key='id'):
        pivot_table = pd.pivot_table(data, index=[index_column], columns=[target_column], values=key, aggfunc='count')

        pivot_table = pivot_table.fillna(0)
        pivot_table["TOTAL"] = pivot_table[1]+pivot_table[0]
        pivot_table['%BAD'] = pivot_table[1]/pivot_table["TOTAL"]
        pivot_table['DIST_BAD'] = pivot_table[1]/pivot_table[1].sum()
        pivot_table['DIST_GOOD'] = pivot_table[0]/pivot_table[0].sum()
        pivot_table['WOE'] = np.log((pivot_table['DIST_BAD'] + 1e-6)/(pivot_table['DIST_GOOD'] + 1e-6))
        pivot_table['IV'] = pivot_table['WOE'] * (pivot_table['DIST_BAD'] - pivot_table['DIST_GOOD'])
        return pivot_table

    @staticmethod
    def calculate_iv(pivot_table, iv='IV'):
        iv = pivot_table[(pivot_table.IV.notnull()) & (pivot_table.IV != np.inf) & (pivot_table.IV  != -np.inf)]['IV'].sum()
        return iv

    @staticmethod
    def generate_woe_mapper(data, index_column, target_column, key='id'):
        pivot_table = Scorecard.calculate_woe(data, index_column, target_column, key)
        return pivot_table[['WOE']].to_dict()['WOE']
    
    def transform_binning_and_pivot_woe(self, col, binning_method='quantile', print_iv=False):
        # determine binning
        self.meta_data[col] = self.binning_builder(self.intime.data[col], binning_method)
            
        # save state
        self.intime_binning[col] = Scorecard.get_binning(self.intime.data[col], self.meta_data[col])

        data = self.intime_binning
        pivot_table = Scorecard.calculate_woe(data, col, self.flag, key=self.key)
        if print_iv:
            print("IV: %2f" % Scorecard.calculate_iv(pivot_table))
        return pivot_table
    
    #SCORECARD GENERATOR
    def load_meta_data_from_dict(self, meta_data):
        self.meta_data = meta_data
    
    def set_binning(self, col, binning_data, evaluate_woe=False, binning_method='quantile'):
        if col not in self.meta_data:
            raise ValueError(f"Column '{col}' not found in meta_data. Please initialize it first.")
        print(col,'bin set into:', binning_data)
        if evaluate_woe:
            print('================ BEFORE Adjustment ================')
            display(self.transform_binning_and_pivot_woe(col, binning_method=binning_method, print_iv=False))
        self.meta_data[col]['binning_data'] = binning_data
        
       
        if evaluate_woe:
            col_type = self.meta_data[col].get('data_type', 'numerical')
            if col_type == 'numerical':
                binned_col = pd.cut(
                    self.intime.data[col],
                    bins=binning_data,
                    include_lowest=True
                )
            else:
                mapping = self.meta_data[col]['binning_data']
                binned_col = self.intime.data[col].astype(str).map(mapping).fillna('UNBINNED')

            temp_df = pd.DataFrame({
                col: binned_col,
                self.flag: self.intime.data[self.flag],
                self.key: self.intime.data[self.key]
            })

            pivot_table = Scorecard.calculate_woe(temp_df, index_column=col, target_column=self.flag, key=self.key)
    
            print('================ After Adjustment ================')
            return pivot_table

    def bin_to_woe(self, data_bin, mapper, col):
        for binning in data_bin.unique():
            mapper.setdefault(binning, self.default_woe)

        data_woe = data_bin.replace(mapper).replace([np.nan, np.inf, -np.inf], self.default_woe)

        if data_woe.isna().any():
            print(f"Warning: Default WOE values used for col {col} because some binnings are not available in mapping")
        if ((data_woe == np.inf) | (data_woe == -np.inf)).any():
            print(f"Warning: Default WOE values used for col {col} because some binnings values are inf")

        return data_woe  

    def woe_transform(self, variables):
        self.intime['data_woe'] = self.intime['data'][[self.key, self.flag]].copy()
        self.oot['data_woe'] = self.oot['data'][[self.key, self.flag]].copy()
        self.intime['data_bin'] = self.intime['data'][[self.key, self.flag]].copy()
        self.oot['data_bin'] = self.oot['data'][[self.key, self.flag]].copy()
        self.intime['mapper'] = {}
        self.variables = variables
        
        cnt_variables = len(self.variables)
        for i, col in enumerate(variables, 1):
#             clear_output(wait=False)
            print(f'Transforming {i}/{cnt_variables} into WoE: {col}')
            if col not in self.meta_data:
                print(f"Binning data for {col} is unavailable, proceeding with optimal binning")
                self.meta_data[col] = self.binning_builder(self.intime.data[col], binning_method='optimal')

            self.intime['data_bin'][col] = self.get_binning(self.intime['data'][col], self.meta_data[col])
            self.oot['data_bin'][col] = self.get_binning(self.oot['data'][col], self.meta_data[col])
            self.intime['mapper'][col] = self.generate_woe_mapper(self.intime['data_bin'], col, self.flag, key=self.key)
            self.intime['data_woe'][col] = self.bin_to_woe(self.intime['data_bin'][col], self.intime['mapper'][col], col)
            self.oot['data_woe'][col] = self.bin_to_woe(self.oot['data_bin'][col], self.intime['mapper'][col], col)
     
    def cv_result(self, model, cv_scoring='roc_auc'):
        cv_scores = cross_val_score(
            estimator=model,
            X=self.intime.data_woe[self.variables],
            y=self.intime.data_woe[self.flag],
            scoring=cv_scoring,
            cv=self.cv
        )

        cv_mean = cv_scores.mean()
        self.intime['evaluation'] = {'CROSS VALIDATION AUC': round(cv_mean, 4)}
        self.oot['evaluation'] = {'CROSS VALIDATION AUC': None}

        print("=" * 60)
        print("Cross-Validation Results (Intime Data)")
        print("-" * 60)
        for i, score in enumerate(cv_scores, 1):
            print(f" Fold {i}: {score:.4f}")
        print("-" * 60)
        print(f" Mean AUC: {cv_mean:.4f}")
        print("=" * 60)

        return cv_mean

    def _generate_pivot_table(self, dataset, col, score=False):
        merged_data = pd.merge(dataset.data_bin, dataset.data_woe[[self.key] + self.variables], on=self.key, suffixes=['', '_WOE'])
        pivot = pd.pivot_table(merged_data, index=[col, col+'_WOE'], values=self.key, columns=[self.flag], aggfunc='count', observed=True)
        pivot["TOTAL"] = pivot[0] + pivot[1]
        pivot["%BAD"] = pivot[1]/pivot["TOTAL"]
        pivot = pivot[["TOTAL","%BAD"]] #Filter columns that are shown in scorecard
        
        if score:
            temp = merged_data[[col, col+'_WOE']].reset_index()
            temp["model_score"] = self.predict_score(self.oot.data.reset_index())
            pivot["median_score_oot"] = temp.groupby([col,col+'_WOE'])['model_score'].median()
        
        return pivot

    def predict_proba(self, data):
        data_bin = pd.DataFrame()
        data_woe = pd.DataFrame()
    
        for col in self.variables:
            if self.meta_data[col]['data_type'] == "categorical":
                data[col] = data[col].astype(str)
                
            data_bin[col] = self.get_binning(data[col], self.meta_data[col])
            mapper = self.intime['mapper'][col]
            data_woe[col] = self.bin_to_woe(data_bin[col], mapper, col)

        return self.model.predict_proba(data_woe)
    
    def train(self, model, params={}, cv_scoring='roc_auc', woe=False, score_card=True):
        self.model = clone(model)
        neg_coef_list = []

        # Run CV and print results
        cv_mean = self.cv_result(self.model, cv_scoring=cv_scoring)
        
        # model training
        self.model.fit(self.intime.data_woe[self.variables], self.intime.data_woe[self.flag])
        coefs = self.model.coef_.ravel()
        
        # scorecard
        self.pivot_tables = {}
        for i, col in enumerate(self.variables):
            print(col)
            intime_pivot = self._generate_pivot_table(self.intime, col,score=False)
            oot_pivot = self._generate_pivot_table(self.oot, col, score=False)

            pivot = intime_pivot.join(oot_pivot, how='left', lsuffix='_intime', rsuffix='_oot')
            pivot = pivot.reset_index()
            
            pivot["SCORE"] = round(coefs[i] * pivot[col+"_WOE"] * self.M * -1)
            self.pivot_tables[col] = pivot
            if coefs[i] < 0:
                print("Negative coefficient observed for %s"%col)
            display(pivot)
        
        # calculate base point
        b0 = self.model.intercept_[0]
        base_point = self.C - b0 * self.M
        self.base_point = base_point
        
        # model evaluation
        y_proba_intime = self.model.predict_proba(self.intime.data_woe[self.variables])[:, 1]
        y_proba_oot = self.model.predict_proba(self.oot.data_woe[self.variables])[:, 1]

        self.intime['evaluation']['AUC'] = roc_auc_score(self.intime.data_woe[self.flag], y_proba_intime)
        self.oot['evaluation']['AUC'] = roc_auc_score(self.oot.data_woe[self.flag], y_proba_oot)

        summary_table = pd.DataFrame({
            'Metric': ['Model Base Point', 'AUC (Intime)','AUC (Cross-Val)', 'AUC (OOT)'],
            'Value': [round(base_point, 3), round(self.intime['evaluation']['AUC'], 4), round(cv_mean, 4),round(self.oot['evaluation']['AUC'], 4)]
        })

        display(summary_table)
     
    @property
    def scorecard_json(self):
        result = {}
        for feature, df in self.pivot_tables.items():
            bin_col = df.columns[0]
            result[feature] = dict(zip(df[bin_col], df['SCORE']))
        return result

    def proba_to_score(self, probabilities):
        score = pd.Series(probabilities).apply(lambda probability: self.C - (self.M * np.log(probability / (1 - probability))))
        return score
    
    def predict_score(self, data):
        proba_bad = self.predict_proba(data[self.variables])[:, 1]
        score = self.proba_to_score(proba_bad)
        return score