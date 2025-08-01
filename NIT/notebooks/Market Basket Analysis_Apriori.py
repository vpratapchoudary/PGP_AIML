import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

def solution():
    data=pd.read_csv('res/Retail.csv')
    #print(data.head(10))
    print('Total data shape:',data.shape)
    print('Unscanned Items shape:',data[data['Dept']=='0999:UNSCANNED ITEMS'].shape)
    data.drop(data.loc[data['Dept']=='0999:UNSCANNED ITEMS'].index, inplace=True)
    print('Data shape after dropping unscanned items:', data.shape)
    res1_candy = (data[data['Dept']=='0973:CANDY'].shape)[0]
    print("number of times ‘0973:CANDY’ sold:",res1_candy)

    #df = data.groupby('POS Txn')
    #print(dataset.head())

    transaction_list=[]
    for i in data['POS Txn'].unique():
        tlist = list(set(data[data['POS Txn']==i]['Dept']))
        if len(tlist)>0:
            transaction_list.append(tlist)

    te = TransactionEncoder()
    te_ary = te.fit(transaction_list).transform(transaction_list)
    df2 = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df2, min_support=0.02, use_colnames = True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=2)
    sup_df = rules.sort_values('support', ascending=False).reset_index()
    res2_maxsupport = round(sup_df['support'][0],5)
    #print(sup_df.iloc[:5,:6])
    #print(sup_df.iloc[:-5,:6])
    print(res2_maxsupport)

    print('Rules shape:', rules.shape)
    res3_totrules = rules.shape[0]
    print(res3_totrules)

    fildf = rules[(rules['lift']>=3) & (rules['confidence']>=0.1)]
    print('Filtered Rules shape:', fildf.shape)
    res4_filrules = fildf.shape[0]
    print(res4_filrules)

    #print(rules)


    # Creating a list of the answer
    result=[res1_candy, res2_maxsupport, res3_totrules, res4_filrules]
    print('Final Result:',result)
    # NOTE: Here 100, 0.54321, 40, 20 are the answer of 1st, 2nd, 3rd and 4th question respectively. Change it accordingly.

    # Finally create a dataframe of the final output  and write the output to output.csv

    result=pd.DataFrame(result)
    # writing output to output.csv
    result.to_csv('output/output.csv', header=False, index=False)