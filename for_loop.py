data=pd.read_csv('/home/alanah/Downloads/tmb_mskcc_2018_clinical_data.tsv',sep='\t')
print(data.head())
#Mutation_Burden = []
for x in data["Mutation Count"]:
    for y in data['Sample coverage']:
        w = ((x/y)*100);
        data['Mutation Burden'] = w

print(data.tail)