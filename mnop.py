from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class changes(BaseEstimator,TransformerMixin):

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        x=X.copy()

        x['Brand']=x['Name'].str.split().str[0]
        x.drop('Name',axis=1,inplace=True)

        x['Mileage']=pd.to_numeric(
            x['Mileage'].astype(str).str.split().str[0],errors="coerce"
        )

        x['Engine']=pd.to_numeric(
            x['Engine'].astype(str).str.split().str[0],errors="coerce"
        )

        x['Power']=pd.to_numeric(
            x['Power'].astype(str).str.split().str[0],errors="coerce"
        )

        return x
