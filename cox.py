import pandas as pd
from lifelines import CoxPHFitter

data=pd.read_csv('/home/alanah/Downloads/tmb_mskcc_2018_clinical_data.tsv',sep='\t')
print(data.head())

data = data.astype({'Overall Survival (Months)':'float'})
print(data['Overall Survival (Months)'])

cph = CoxPHFitter()
cph.fit(data, duration_col='Overall Survival (Months)', event_col='Overall Survival Status')

cph.print_summary()

