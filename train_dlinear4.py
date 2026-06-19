import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import IPython
import torch.utils.data as data
import torch
import dlinear_models as models
import torch.nn as nn
from tqdm import tqdm


import IPython
ipython = IPython.get_ipython()
if ipython is not None:
    ipython.run_line_magic(magic_name="load_ext", line="autoreload")
    ipython.run_line_magic(magic_name="autoreload", line="2")
else:
    print("Autoreload enabled in IPython.")
    print("Not running in IPython; autoreload not enabled.")



def generate_data(num_steps, interval = 0.1):
    x = np.linspace(0, num_steps * interval, num_steps)
    y = np.sin(x) +np.sin(3*x)*0.4 + np.random.normal(0, 0.1, x.shape) + 0.1*x
    return y

class datanorm:
    def __init__(self,inp_len,seq_len,norm=[None,None]):

        self.inp_len = inp_len
        self.seq_len = seq_len
        self.norm = norm[0]
        self.params = [norm[1]] 


    def norm_data(self,data,other):
        if self.norm == "none":
            return data,other
        if self.norm == "subtract_first":
            subtr = data[:,0:1,:]
            data = (data-subtr)*0.1
            other = (other-subtr)*0.1
            self.params = subtr
            return data,other

        if self.norm == "mean_std":
            mean = torch.mean(data[:,:self.seq_len,:],dim=1,keepdim=True)
            std = torch.std(data[:,:self.seq_len,:],dim=1,keepdim=True)
            #print("mean,std:",mean.shape,std.shape)
            data = (data-mean)/std*0.1
            other = (other-mean)/std*0.1
            self.params = [mean,std]
            return data,other

        if self.norm == "max_divide":
            return data/self.params[0],other/self.params[0]

    def denorm_data(self,data):
        if self.norm == "none":
            return data

        if self.norm == "subtract_first":
            data = data/0.1+self.params
            return data
        if self.norm == "mean_std":
            data = data*self.params[1]/0.1+self.params[0]
            return data
        if self.norm == "max_divide":
            return data*self.params[0]




class TrafficDataset(data.Dataset):
    def __init__(self, path, junction, seq_len, pred_len, mode="train"):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        dataset = pd.read_csv(path)
        data = dataset.copy()
        data["DateTime"] = pd.to_datetime(data["DateTime"])
        
        df = data.copy() 
        df["Month"] = df['DateTime'].dt.month
        df["Date_no"] = df['DateTime'].dt.day
        df["Weekday"] = df['DateTime'].dt.weekday
        df["Hour"] = df['DateTime'].dt.hour

        sel = np.where(df["Junction"] == junction)[0]

        self.data = df["Vehicles"].values[sel]
        self.data_dateno = df["Date_no"].values[sel]
        self.data_month = df["Month"].values[sel]   
        self.data_day = df["Weekday"].values[sel]
        self.data_hour = df["Hour"].values[sel]

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.data_dateno = torch.tensor(self.data_dateno, dtype=torch.long)
        self.data_month = torch.tensor(self.data_month, dtype=torch.long)
        self.data_day = torch.tensor(self.data_day, dtype=torch.long)
        self.data_hour = torch.tensor(self.data_hour, dtype=torch.long)

        total_len = len(self.data)
        train_end = int(total_len * 0.8)
        
        if mode == "train":
            self.data_idx = torch.arange(0, train_end).long()
        else:
            self.data_idx = torch.arange(train_end, total_len - 1 - seq_len - pred_len).long()

    def __getitem__(self, idx_):
        idx = self.data_idx[idx_]
        start = idx
        end = idx + self.seq_len + self.pred_len

        tokens = (self.data[start:end].reshape(1, -1)).clone().float()

        times = torch.cat([
            self.data_month[start:end].unsqueeze(0),
            self.data_dateno[start:end].unsqueeze(0),
            self.data_day[start:end].unsqueeze(0),
            self.data_hour[start:end].unsqueeze(0)
        ], dim=0)

        times = times.permute(1, 0)
        tokens = tokens.float().permute(1, 0)
        
        return (
            tokens[:self.seq_len], 
            tokens[self.seq_len-1:-1], 
            tokens[self.seq_len:], 
            times[:self.seq_len], 
            times[self.seq_len-1:-1]
        )

    def __len__(self):
        return len(self.data_idx)



class ETTDataset(data.Dataset):
    def __init__(self, path, seq_len, pred_len, mode="train", target_col="OT"):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        dataset = pd.read_csv(path)
        data = dataset.copy()
        
        data["date"] = pd.to_datetime(data["date"])
        
        df = data.copy() 
        df["Year"] = df['date'].dt.year
        df["Month"] = df['date'].dt.month
        df["Date_no"] = df['date'].dt.day
        df["Weekday"] = df['date'].dt.weekday
        df["Hour"] = df['date'].dt.hour
        
        self.data = df[target_col].values
        
        self.data_dateno = df["Date_no"].values
        self.data_month = df["Month"].values   
        self.data_day = df["Weekday"].values
        self.data_hour = df["Hour"].values

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.data_dateno = torch.tensor(self.data_dateno, dtype=torch.long)
        self.data_month = torch.tensor(self.data_month, dtype=torch.long)
        self.data_day = torch.tensor(self.data_day, dtype=torch.long)
        self.data_hour = torch.tensor(self.data_hour, dtype=torch.long)

        total_len = len(self.data)
        train_end = int(total_len * 0.8)
        
        if mode == "train":
            self.data_idx = torch.arange(0, train_end).long()
        else:
            self.data_idx = torch.arange(train_end, total_len - 1 - seq_len - pred_len).long()

    def __getitem__(self, idx_):
        idx = self.data_idx[idx_]
        start = idx
        end = idx + self.seq_len + self.pred_len

        tokens = (self.data[start:end].reshape(1, -1)).clone().float()
        
        times = torch.cat([
            self.data_month[start:end].unsqueeze(0),
            self.data_dateno[start:end].unsqueeze(0),
            self.data_day[start:end].unsqueeze(0),
            self.data_hour[start:end].unsqueeze(0)
        ], dim=0)

        times = times.permute(1, 0)
        tokens = tokens.float().permute(1, 0)
        
        return (
            tokens[:self.seq_len], 
            tokens[self.seq_len-1:-1], 
            tokens[self.seq_len:], 
            times[:self.seq_len], 
            times[self.seq_len-1:-1]
        )

    def __len__(self):
        return len(self.data_idx)
    



def process_dataset(loader,mode,lr_scheduler=None):
    stat = {"loss":0,"acc":0,"tot":0}
    print("### PROCESSING MODE:",mode)
    for pos,data in enumerate(tqdm(iter(loader))):
        enc_inp,dec_inp,tgt,times_src,times_tgt = data


        if pos==0:
            print("\n")
            print("test:",enc_inp[0,-3:],dec_inp[0,:3])
            print("\n")

        #loader.dataset.pred_len = 24*4
        if mode == "train":
            #loader.dataset.pred_len = np.random.randint(1,5)*24
            optimizer.zero_grad()
            enc_inp = enc_inp.to(device)
            dec_inp = dec_inp.to(device)
            tgt = tgt.to(device)
            times_src = times_src.to(device)
            times_tgt = times_tgt.to(device)
            enc_inp,dec_inp = normalizer.norm_data(enc_inp,dec_inp)

            out = model(enc_inp,dec_inp)
            out = normalizer.denorm_data(out)
            loss = criterion(out,tgt)/(torch.var(tgt) + 1e-5)
            loss.backward()
            optimizer.step()
            if not lr_scheduler is None:
                lr_scheduler.step()

        if mode == "test":
            with torch.no_grad():
                enc_inp = enc_inp.to(device)
                dec_inp = dec_inp.to(device)
                tgt = tgt.to(device)
                times_src = times_src.to(device)
                times_tgt = times_tgt.to(device)

                enc_inp,dec_inp = normalizer.norm_data(enc_inp,dec_inp)
                
                out = model(enc_inp,dec_inp)
                if pos==1:
                    print("\n out:",out[0][:3])
                out = normalizer.denorm_data(out)
                loss = criterion(out,tgt)/(torch.var(tgt) + 1e-5)

        if mode == "infer":
            with torch.no_grad():
                enc_inp = enc_inp.to(device)
                dec_inp = dec_inp.to(device)*0
                #print("enc,dec,pred:",enc_inp.shape,dec_inp.shape,tgt.shape)

                enc_inp,dec_inp = normalizer.norm_data(enc_inp,dec_inp)


                tgt = tgt.to(device)
                times_src = times_src.to(device)
                times_tgt = times_tgt.to(device)
                pred_len = loader.dataset.pred_len
                dec_inp = enc_inp[:,-1:].clone()
                #dec_inp[:,0]=enc_inp[:,-1].clone()

                if model_type.find("Transformer")>-1 or model_type == "DLinear_MHE":
                    for i in range(pred_len):
                        out = model(enc_inp,dec_inp)
                        #if i<10:
                        #    print("i:",i,"out:",out[0][:15],"dec_inp:",dec_inp[0][:15])
                        dec_inp = torch.cat([dec_inp,out[:,-1:]],dim=1)
                        #dec_inp[:,i+1]=out[:,i]

                    dec_inp = dec_inp[:,1:]
                    out = dec_inp
                if pos==1:
                    print("\n ###")
                    print("\n eval out:",out[0][:3])


                if model_type == "DLinear" or model_type == "DLinear_trend":
                    enc_inp = enc_inp[:,:hist_len]
                    out = model(enc_inp,None)
                    dec_inp=out


                dec_inp = normalizer.denorm_data(dec_inp)
                out = normalizer.denorm_data(out.detach())

                loss = criterion(out,tgt)/(torch.var(tgt) + 1e-5)


        acc = loss.detach().cpu() 

        if pos==1 and pos>0:
            print("pos:",pos,"loss:",loss.item(),"acc:",acc.item())
            plt.figure()
            plt.plot(out[0].detach().cpu().reshape(-1),"r-")
            plt.plot(tgt[0].detach().cpu().reshape(-1),"b-")
            try:
                plt.plot(model.trend_out[0].detach().cpu().reshape(-1),"m-")
            except:
                pass

            plt.title("Mode:"+str(mode)+" Model:"+model_type)
            # plt.show()
            plt.savefig(f"{model_type}_traffic_plot_{mode}33.png")
            plt.close()

            if mode == "infer":
                break


        stat["loss"]+=loss.item()
        stat["acc"]+=acc.item()
        stat["tot"]+=1

    stat["loss"]=stat["loss"]/stat["tot"]
    stat["acc"]=stat["acc"]/stat["tot"]

    return stat

def model_scheduler(cfg,epoch,loss,mode="save"):
    import json
    import copy

    if mode == "restore":
        cfg["model"].load_state_dict(torch.load(cfg["spath"],weights_only=False,map_location="cpu"))
        cfg["model"]=cfg["model"].to(cfg["device"])
        return cfg["model"]

    
    cfg["loss"]+=[loss]
    if cfg["best_loss"] is None:
        cfg["best_loss"] = loss
    if  cfg["best_loss"]>=loss:
        print("Found better solution!")
        cfg["best_loss"]=loss
        cfg["model"].eval()
        torch.save(cfg["model"].state_dict(),cfg["spath"])
        cfg["model"].train()

        json_path = cfg["spath"][:cfg["spath"].rfind("."):]+".json"

        with open(json_path,"w") as f:
            cfg2 = copy.copy(cfg)
            cfg2["model"]=None
            json.dump(cfg2,f)



if __name__ == "__main__":
    import sys

    hist_len = 720
    pred_len = 480

    data_norm = "none"
    if len(sys.argv)>2:
        data_norm = sys.argv[2] 
    
    dataset_train = TrafficDataset("traffic.csv", junction=1, seq_len=hist_len, pred_len=pred_len, mode="train")
    train_loader = data.DataLoader(dataset_train, batch_size=8, shuffle=True) 

    dataset_test = TrafficDataset("traffic.csv", junction=1, seq_len=hist_len, pred_len=pred_len, mode="test")
    test_loader = data.DataLoader(dataset_test, batch_size=8, shuffle=False)

    normalizer = datanorm(hist_len,pred_len,[data_norm,torch.max(dataset_train.data)])


    batch = next(iter(train_loader))
    print("test batch shape:",batch[0].shape)
    enc_inp,dec_inp = normalizer.norm_data(batch[0],batch[1]) 
    #sysexit(0)


    model_type = sys.argv[1]



    if model_type == "DLinear":
        cfg={"input_dim":1,"embed_dim":256,"out_dim":1,"num_layers":2,"device":"cuda:0","inp_seq_len":hist_len,"outp_seq_len":pred_len}
    if model_type == "DLinear_MHE":
        cfg={"input_dim":1,"embed_dim":16,"out_dim":1,"num_layers":1,"device":"cuda:0","inp_seq_len":hist_len,"outp_seq_len":pred_len}
    if model_type == "DLinear_trend":
        cfg={"input_dim":1,"embed_dim":hist_len,"out_dim":1,"num_layers":1,"device":"cuda:0","inp_seq_len":hist_len,"outp_seq_len":pred_len}


    if model_type == "Transformer_base" or model_type == "Transformer_base_detailed":
        cfg={"input_dim":1,"embed_dim":128,"num_heads":8,"out_dim":1,"enc_numlayers":1,"dec_numlayers":1,"dropout":0.1,"temporal_encoding":False,"device":"cuda:0","inp_seq_len":hist_len,"outp_seq_len":pred_len}
    if model_type == "Transformer_MT_S1":
        cfg={"input_dim":1,"input_slice":24,"embed_dim":128,"num_heads":8,"out_dim":1,"enc_numlayers":1,"dec_numlayers":1,"dropout":0.1,"temporal_encoding":False,"device":"cuda:0","inp_seq_len":hist_len,"outp_seq_len":pred_len}


    if model_type == "Transformer_MT_VariableStep":
        cfg={"input_dim":1, "input_slice":48, "stride":12, "embed_dim":128,"num_heads":8,"out_dim":1,"enc_numlayers":1,"dec_numlayers":1,"dropout":0.1,"temporal_encoding":False,"device":"cuda:0","inp_seq_len":hist_len,"outp_seq_len":pred_len}



    if model_type == "Transformer_MT_S1_decode":
        cfg={"input_dim":1,"input_slice":24,"embed_dim":128,"num_heads":8,"out_dim":1,"dec_numlayers":1,"dropout":0.1,"temporal_encoding":False,"device":"cuda:0","inp_seq_len":hist_len,"outp_seq_len":pred_len}

    if model_type == "Transformer_MT_S24":
        cfg={"input_dim":1,"input_slice":24,"embed_dim":128,"num_heads":8,"out_dim":1,"enc_numlayers":1,"dec_numlayers":1,"dropout":0.1,"temporal_encoding":False,"device":"cuda:0","inp_seq_len":hist_len,"outp_seq_len":pred_len}
    if model_type == "Transformer_trend":
        cfg={"input_dim":1,"embed_dim":128,"num_heads":8,"out_dim":1,"enc_numlayers":2,"dec_numlayers":2,"dropout":0.1,"temporal_encoding":False,"device":"cuda:0","inp_seq_len":hist_len,"outp_seq_len":pred_len}



    if model_type == "Transformer_EncoderFree":
        cfg={
            "input_dim": 1,"embed_dim": 128,"num_heads": 8,"out_dim": 1,"dec_numlayers": 2,"dropout": 0.1,"temporal_encoding": False,"device": "cuda:0","inp_seq_len": hist_len,"outp_seq_len": pred_len}



    cfg["hist_len"] = hist_len
    cfg["pred_len"] = pred_len
    cfg["norm_type"] = data_norm


    cfg["data_norm"] = data_norm

    device = "cuda:0"

    if model_type == "Transformer_base":
        model = models.Transformer_base(cfg)

    if model_type == "Transformer_base_detailed":
        model = models.Transformer_base_detailed(cfg)

    if model_type == "DLinear":
        model = models.DLinear(cfg)
    if model_type == "DLinear_MHE":
        model = models.DLinear_MHE(cfg)
    if model_type == "DLinear_trend":
        model = models.DLinear_trend(cfg)


    if model_type == "Transformer_MT_S1":
        model = models.Transformer_MT_S1(cfg)


    if model_type == "Transformer_MT_VariableStep":
        model = models.Transformer_MT_VariableStep(cfg)


    if model_type == "Transformer_MT_S1_decode":
        model = models.Transformer_MT_S1_decode(cfg)


    if model_type == "Transformer_MT_S24":
        model = models.Transformer_MT_S24(cfg)
    if model_type == "Transformer_trend":
        model = models.Transformer_trend(cfg)

    if model_type == "Transformer_EncoderFree":
        model = models.Transformer_EncoderFree(cfg)

    out = model(batch[0],batch[1])
    print("out:",out.shape)


    model = model.to(device)
    #define optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4) 

    #learning rate scheduler
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=100,T_mult=1,eta_min=0.)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 1e-5, max_lr=1e-3, step_size_up=20, mode="triangular", cycle_momentum=False)


    criterion = nn.MSELoss()

    train_schedule = {"loss":[],"best_loss":None,"spath":"runs/"+model_type+"_traffic_train.pytorch","model":model,"device":device}
    test_schedule = {"loss":[],"best_loss":None,"spath":"runs/"+model_type+"_traffic_test.pytorch","model":model,"device":device}



    if 1==1:
        for epoch in range(50):
            model.train()
            train_stat = process_dataset(train_loader,"train")
            print("### epoch:",epoch,"train stat:",train_stat)
            model_scheduler(train_schedule,epoch,train_stat["loss"])

            model.eval()
            test_stat = process_dataset(test_loader,"test")
            print("### epoch:",epoch,"test stat:",test_stat)
            model_scheduler(test_schedule,epoch,test_stat["loss"])
            model.eval()
            eval_stat = process_dataset(test_loader,"infer")
            print("### epoch:",epoch,"infer stat:",eval_stat)


print("Best rez on test:",test_schedule["best_loss"])

epoch = -1
print("### evaluation on best test rez:")
model = model_scheduler(test_schedule,epoch,None,mode="restore")
model.eval()
eval_stat = process_dataset(test_loader,"infer")
print("\n")
print("Final eval stat:",eval_stat)
