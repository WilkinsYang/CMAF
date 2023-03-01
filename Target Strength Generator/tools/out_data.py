import pandas as pd

def out_data(df):
    list=[]
    for index,row in df.iterrows():
        list.append([row['ID'],(row['Maxlength'])*1500/(637.16*2),row['TS'],row['location_x']*1500/637.16,row['location_y']*1500/637.16,1,row['file'],
                        row['x'], row['y'], row['w'], row['h']])

    df2=pd.DataFrame(list,columns=['ID','average(cm)','TS','locx','locy','num','file','x','y','w','h'])
    df2=df2.sort_values(by=['file'])
    return df2




