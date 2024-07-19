import pandas as pd
import numpy as np
import subprocess
import warnings
import requests
import sys
import os

def SFP(date:str ,length:int = 30,freqs = ["410MHz","610MHz","1415MHz","2695MHz"], mode:str = "all",method:str="norm",poww=1/2):
    length+=1
    sform = "%y%m%d"
    fcols = ["from 410MHz","from 610MHz","from 1415MHz","from 2695MHz"]
    
    if mode == "none":
        return SFQ(date,freqs,sform=sform).iloc[0]
    
    for date_i in pd.date_range(end=(pd.to_datetime(date, format=sform)), periods=length).strftime(sform):
        df = SFQ(date_i,freqs,sform=sform)
        try: 
            sf = pd.concat([sf,df])
        except:
            sf = df
    
    pastdata = sf.copy()
    pastdata.drop(pastdata.tail(1).index,inplace=True)
    past_average = pastdata.mean()
    normalized = sf/past_average
    for freq in freqs:
        rescaled = rescale(normalized, average = past_average, freq=freq,method=method,poww=poww)
        last2d_error,error_stat = cerror(rescaled, freq,freqs,sf,fcols)
        try:
            tb = pd.concat([tb,rescaled.tail(1)])
            er = pd.concat([er,last2d_error],axis=1)
            st = pd.concat([st,error_stat],axis=1)
        except:
            tb = rescaled.tail(1)
            er = last2d_error.copy()
            st = error_stat.copy()
    tb.columns = fcols
    tb = tb.transpose()
    tb.columns = freqs
    tb = pd.concat([sf.tail(1),tb])
    op0, op1, op2 = tb, er, st
    op0 = op0.astype(int).astype(float)
    op1 = op1.map('{:.2f}%'.format)
    op2 = op2.map('{:.2f}%'.format)
    del tb, er, st
    
    match mode:
        case "all":
            return op0,op1,op2
        case "410MHz":
            return op0.loc[["from 410MHz"]].iloc[0]
        case "610MHz":
            return op0.loc[["from 610MHz"]]
        case "1415MHz":
            return op0.loc[["from 1415MHz"]]
        case "2695MHz":
            return op0.loc[["from 2695MHz"]]

def SFQ(date, freqs=["410MHz","610MHz","1415MHz","2695MHz"], path="/data/analysis/solar/",sform = "%y%m%d", debug=False):
   
    path = makefile(path)
    url = "https://downloads.sws.bom.gov.au/wdc/wdc_solradio/data/learmonth/SRD/"
    #path = "/home/thchenlin/solarfluxdata/"
    emptyfile = "reduced/emptyfile.SRD"
    
    lform, tz_local,tz_server = "%y%m%d%H%M%S", "Asia/Taipei", "UTC"
    columns={0: 'Time', 1: '245MHz', 2: '410MHz', 3:'610MHz',4: '1415MHz', 5: '2695MHz', 6:'4995MHz',7: '8800MHz', 8:'15400MHz'}
    
    #set sunrise and sunset with local tz data
    sunrise_local = (pd.to_datetime(date+"050000", format=lform)).tz_localize(tz_local)
    sunset_local = (pd.to_datetime(date+"190000", format=lform)).tz_localize(tz_local)
    
    #change sunrise and sunset to local tz
    sunrise_server = sunrise_local.tz_convert(tz_server)
    sunset_server = sunset_local.tz_convert(tz_server)
    dates = [sunrise_server]
    if sunrise_server.date() != sunset_server.date(): dates.append(sunset_server)
    
    #create list of to-be-used columns
    freqt = freqs.copy() #copy to-be-used freq
    freqt.insert(0,"Time") #add "Time" at beginning
    invcol = {v: k for k, v in columns.items()} #create inverse dict of server columns
    newcol = {k: invcol[k] for k in freqt if k in invcol} #filter columns
    columns1 = {v: k for k, v in newcol.items()} #re-inverse back to-be-used columns
    usecol = list(columns1.keys()) #convert to list

    #check in reduced database
    rfilename = sunrise_local.strftime("RL%y%m%d.SRD")
    if not subprocess.run(["test","-f", path+"reduced/"+rfilename]).returncode:
        dl = (pd.read_csv(path+"reduced/"+rfilename, sep=",",engine='python', usecols = usecol,index_col=0).squeeze()).astype(float)
        dl = dl.rename(str(dl.name))
        if debug: print(f"Reduced data \"{rfilename}\" exists")
        return sanity_test(dl) ## exit if exists
    
    for daydate in dates: ###
        filename = daydate.strftime("L%y%m%d.SRD")
        existence = not subprocess.run(["test","-f", path+filename]).returncode
        exception = False
        if existence == 0: #if file not downloaded
            fileurl = url + daydate.strftime("%Y/") + filename
            webtrue = False if (requests.head(fileurl).status_code) == 404 else True
            if webtrue: #if exists in database
                existence, exception = web_dl(path, fileurl, exception)
                
        if not exception:    #if no error   
            filefpath = path+filename if existence == 1 else path+emptyfile
            df = loadffile(filefpath,columns,sform,lform,tz_server,tz_local,daydate)

            try:
                dz = pd.concat([dz, df])
            except:
                dz = df
   
    res = cleanup(dz, sunrise_local, sunset_local, date)
    narr = []
    [narr.append(str(val)) if str(val) == "nan" else narr.append(str(val)[:-2]) for val in res]
    ttext = f'{date},{",".join(narr)}\n'

    with open(path+"reduced/"+rfilename,"x") as new:
        header = f'Date,{",".join(list(columns.values())[1::])}\n'
        new.write(header+ttext)
    
    return sanity_test(res[freqs])

def rescale(normalized, average, freq,method="sqrt",poww = 1/2):
    match method:
        case "sqrt":
            rescaled = normalized.apply(np.sqrt)*average[freq]
        case "norm":
            rescaled = normalized*average[freq]
        case "root":
            print("In Development, replacing with \"sqrt\"")
            rescaled = normalized.apply(np.sqrt)*average[freq]
            
    return rescaled

def cerror(rescaled, freq,freqs,sf,fcols):
    error = (rescaled.copy())
    for f in freqs:
        error[f] = ((sf[freq]-error[f]).abs())/sf[freq]*100
    error_mean = error.mean()
    error_std = error.std()
    error_stat = pd.concat([error_mean,error_std],axis=1)
    error_stat = error_stat.transpose()
    error_stat.columns = fcols
    error_stat = error_stat.transpose()
    statcol = [f"{freq} Error% Mean", f"{freq} Error% STD"]
    error_stat.columns = statcol
    last2d_error = (error.tail(2)).copy()
    last2d_error.columns = fcols
    l2de = last2d_error.transpose()
    l2de.columns = [f'{freq} d-1 error%', f'{freq} d error%']
    return l2de,  error_stat

def sanity_test(data):
    df = (pd.DataFrame(data)).transpose()
    #df.append(data)
    df["410MHz"] = np.nan if data["410MHz"] < 0.25*(10**6) else data["410MHz"]
    df["610MHz"] = np.nan if data["610MHz"] < 0.40*(10**6) else data["610MHz"]
    df["1415MHz"] = np.nan if data["1415MHz"] < 0.55*(10**6) else data["1415MHz"]
    df["2695MHz"] = np.nan if data["2695MHz"] < 0.80*(10**6) else data["2695MHz"]
    return df

def cleanup(dz,sunrise_local,sunset_local,date):
    dz = dz.resample("1s").mean()
    dz = dz.loc[sunrise_local:sunset_local]
    dz.where((dz < 600000) & (dz > 0) ,np.nan,inplace=True)
    res = dz.median()*10000
    res = res.rename(date)
    return res

def web_dl(path, fileurl, exception):
    try:
        subprocess.run(["wget","-q", "-P", path, fileurl], check=True)
        existence = 1 #set "downloaded" to True  
    except:
        exception = True
        print("Unknown Error", url)
    return existence, exception

def loadffile(filefpath,columns,sform,lform,tz_server,tz_local,daydate):
    df = pd.read_csv(filefpath, sep=" ",engine='python',header=None,dtype='str')
    pd.set_option('future.no_silent_downcasting', True)
    df = df.replace("//////",np.nan)
    df = df.rename(columns=columns)
    df['Time']=df['Time'].apply(lambda x: x[:6])
    df['Time']=daydate.strftime(sform) + df['Time'].astype(str)
    df['Time'] = (pd.to_datetime(df.Time, format=lform))
    df = (df.set_index('Time')).tz_localize(tz_server)
    df = df.tz_convert(tz_local)
    df = df.astype("float")
    return df

def makefile(path):
    path1 = path
    if not os.path.exists(path + "original"): 
        try:
            os.makedirs(path + "original") 
        except:
            path1 = "solar/"
            if not os.path.exists("solar/original"): 
                os.makedirs("solar/original")
                
    
    if not os.path.exists(path + "reduced"): 
        try:
            os.makedirs(path + "reduced") 
        except:
            path1 = "solar/"
            if not os.path.exists("solar/reduced"): 
                os.makedirs("solar/reduced")
                print("reduced created")
                
                
    if not os.path.isfile(path1 + "reduced/emptyfile.SRD"):
        with open(path1+"reduced/emptyfile.SRD","x") as empty:
            empty.write("000000 ////// ////// ////// ////// ////// ////// ////// //////")
            
    return path1



def main(date):
    op0, op1, op2 = SFP(date)
    return op0, op1, op2

if __name__ == "__main__":
    date = sys.argv[1]
    op0,op1,op2 = main(date)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(op0)
        print(op1)
        print(op2)
