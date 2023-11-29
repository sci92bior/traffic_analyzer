import glob

import pandas as pd
import numpy as np


class FeaturesConverter:
    def __init__(self, dataframe_path):

        csv_list = glob.glob(f"{dataframe_path}/*.csv")
        dataframes_normal = [pd.read_csv(i) for i in csv_list if "normal" in i]
        dataframes_anormal = [pd.read_csv(i) for i in csv_list if "normal" not in i]
        for df in dataframes_normal:
            df['Label'] = 0
        for df in dataframes_anormal:
            df['Label'] = 1
        self.old_df = pd.concat(dataframes_normal+dataframes_anormal, ignore_index=True)
        #self.old_df = self.old_df[self.old_df['Etype'] != 2054]

        self.new_df = pd.DataFrame()

    def process_flow_id(self, column_label="Flow ID"):
        self.new_df[column_label] = self.old_df[column_label]
        # TODO

    def process_ip(self, column_label):
        labels = [f"{column_label}_{i}" for i in range(4)]
        self.new_df[labels] = (
                                      self.old_df[column_label].str.split(".", expand=True).astype(float) / 255
                              )

    def divide_by_value(self, column_label, value):
        self.new_df[column_label] = (
                                            self.old_df[column_label].astype(float) / value
                                    )

    def process_one_hot(self, column_label):
        labels = [f"{column_label}_{i}" for i in self.old_df[column_label].unique()]
        print(labels)
        self.new_df[labels] = (
                pd.get_dummies(self.old_df[column_label], columns=labels)
        )

    def process_time(self, column_label):
        temp = self.old_df[column_label].str.split(" ", expand=True)
        pd.to_datetime(temp.col2, format="%Y%m%d%H%M%S")
        temp[0].str.split("/", expand=True)
        # TODO

    def abs_old(self, column_label):
        self.old_df[column_label] = self.old_df[column_label].abs()

    def normalize_through_rows(self, column_label):
        """to [0,1]"""
        #print("Min")
        #print(self.old_df[column_label].min())
        #print(self.old_df[column_label].max())
        self.new_df[column_label] = (
                                            (self.old_df[column_label] - self.old_df[column_label].min())
                                            / (self.old_df[column_label].max() - self.old_df[column_label].min())
                                    )

    def std_through_rows(self, column_label):
        """to mean 0, std 1"""
        self.new_df[column_label] = (
                                            self.old_df[column_label] - self.old_df[column_label].mean()
                                    ) / (self.old_df[column_label].std())

    def process_label(self, column_label):
        # SWAP normal to 0
        self.uni = list(self.old_df[column_label].unique())
        normal_index = self.uni.index("Normal")
        temp = self.uni[0]
        self.uni[0] = self.uni[normal_index]
        self.uni[normal_index] = temp
        nums = [i for i in range(len(self.uni))]

        self.new_df[column_label] = self.old_df[column_label].replace(self.uni,nums)

    def sigmoid(self, column_label):
        self.new_df[column_label] = 1/(1+np.exp(-self.old_df[column_label]))
    def half_sigmoid(self, column_label):
        self.new_df[column_label] = ((1 / (1 + np.exp(-self.old_df[column_label])))-0.5)*2


    def test(self, column_label):
        print(self.old_df[column_label].max())
        print(self.old_df[column_label].min())
        uni = self.old_df[column_label].unique()
        print(len(uni))
        uni.sort()
        print(uni)

    def process_data(self, config, dataset):
        if dataset == "SDN":
            config = config["SDN"]
            if config["Flow ID"]:
                self.process_flow_id("Flow ID")
            if config["Src IP"]:
                self.process_ip("Src IP")
            if config["Src Port"]:
                self.divide_by_value("Src Port", 65535)
            if config["Dst IP"]:
                self.process_ip("Dst IP")
            if config["Dst Port"]:
                self.divide_by_value("Dst Port", 65535)
            if config["Protocol"]:
                self.process_one_hot("Protocol")
            if config["Timestamp"]:
                self.process_time("Timestamp")
            if config["Flow Duration"]:
                self.half_sigmoid("Flow Duration")
            if config["Tot Fwd Pkts"]:
                self.half_sigmoid("Tot Fwd Pkts")
            if config["Tot Bwd Pkts"]:
                self.half_sigmoid("Tot Bwd Pkts")
            if config["TotLen Fwd Pkts"]:
                self.half_sigmoid("TotLen Fwd Pkts")
            if config["TotLen Bwd Pkts"]:
                self.half_sigmoid("TotLen Bwd Pkts")
            if config["Fwd Pkt Len Max"]:
                self.divide_by_value("Fwd Pkt Len Max", 65535)
            if config["Fwd Pkt Len Min"]:
                self.divide_by_value("Fwd Pkt Len Min", 65535)
            if config["Fwd Pkt Len Mean"]:
                self.divide_by_value("Fwd Pkt Len Mean", 65535)
            if config["Fwd Pkt Len Std"]:
                self.divide_by_value("Fwd Pkt Len Std", 65535)
            if config["Bwd Pkt Len Max"]:
                self.divide_by_value("Fwd Pkt Len Max", 65535)
            if config["Bwd Pkt Len Min"]:
                self.divide_by_value("Fwd Pkt Len Min", 65535)
            if config["Bwd Pkt Len Mean"]:
                self.divide_by_value("Fwd Pkt Len Mean", 65535)
            if config["Bwd Pkt Len Std"]:
                self.divide_by_value("Fwd Pkt Len Std", 65535)
            if config["Flow Byts/s"]:
                self.half_sigmoid("Flow Byts/s")
            if config["Flow Pkts/s"]:
                self.half_sigmoid("Flow Pkts/s")
            if config["Flow IAT Mean"]:
                self.half_sigmoid("Flow IAT Mean")
            if config["Flow IAT Std"]:
                self.half_sigmoid("Flow IAT Std")
            if config["Flow IAT Max"]:
                self.half_sigmoid("Flow IAT Max")
            if config["Flow IAT Min"]:
                self.half_sigmoid("Flow IAT Min")
            if config["Fwd IAT Tot"]:
                self.half_sigmoid("Fwd IAT Tot")
            if config["Fwd IAT Mean"]:
                self.half_sigmoid("Fwd IAT Mean")
            if config["Fwd IAT Std"]:
                self.half_sigmoid("Fwd IAT Std")
            if config["Fwd IAT Max"]:
                self.half_sigmoid("Fwd IAT Max")
            if config["Fwd IAT Min"]:
                self.half_sigmoid("Fwd IAT Min")
            if config["Bwd IAT Tot"]:
                self.half_sigmoid("Bwd IAT Tot")
            if config["Bwd IAT Mean"]:
                self.half_sigmoid("Bwd IAT Mean")
            if config["Bwd IAT Std"]:
                self.half_sigmoid("Bwd IAT Std")
            if config["Bwd IAT Max"]:
                self.half_sigmoid("Bwd IAT Max")
            if config["Bwd IAT Min"]:
                self.half_sigmoid("Bwd IAT Min")
            if config["Fwd PSH Flags"]:
                self.new_df["Fwd PSH Flags"] = self.old_df["Fwd PSH Flags"]  # 0
            if config["Bwd PSH Flags"]:
                self.new_df["Bwd PSH Flags"] = self.old_df["Bwd PSH Flags"]
            if config["Fwd URG Flags"]:
                self.new_df["Fwd URG Flags"] = self.old_df["Fwd URG Flags"]  # 0
            if config["Bwd URG Flags"]:
                self.new_df["Bwd URG Flags"] = self.old_df["Bwd URG Flags"]
            if config["Fwd Header Len"]:
                self.half_sigmoid("Fwd Header Len")
            if config["Bwd Header Len"]:
                self.half_sigmoid("Bwd Header Len")
            if config["Fwd Pkts/s"]:
                self.half_sigmoid("Fwd Pkts/s")
            if config["Bwd Pkts/s"]:
                self.half_sigmoid("Bwd Pkts/s")
            if config["Pkt Len Min"]:
                self.divide_by_value("Pkt Len Min", 65535)
            if config["Pkt Len Max"]:
                self.divide_by_value("Pkt Len Max", 65535)
            if config["Pkt Len Mean"]:
                self.divide_by_value("Pkt Len Mean", 65535)
            if config["Pkt Len Std"]:
                self.divide_by_value("Pkt Len Std", 65535)
            if config["Pkt Len Var"]:
                self.half_sigmoid("Pkt Len Var")
            if config["FIN Flag Cnt"]:
                self.new_df["FIN Flag Cnt"] = self.old_df["FIN Flag Cnt"]
            if config["SYN Flag Cnt"]:
                self.new_df["SYN Flag Cnt"] = self.old_df["SYN Flag Cnt"]
            if config["RST Flag Cnt"]:
                self.new_df["RST Flag Cnt"] = self.old_df["RST Flag Cnt"]
            if config["PSH Flag Cnt"]:
                self.new_df["PSH Flag Cnt"] = self.old_df["PSH Flag Cnt"]
            if config["ACK Flag Cnt"]:
                self.new_df["ACK Flag Cnt"] = self.old_df["ACK Flag Cnt"]
            if config["URG Flag Cnt"]:
                self.new_df["URG Flag Cnt"] = self.old_df["URG Flag Cnt"]
            if config["CWE Flag Count"]:
                self.new_df["CWE Flag Count"] = self.old_df["CWE Flag Count"]  # 0
            if config["ECE Flag Cnt"]:
                self.new_df["ECE Flag Cnt"] = self.old_df["ECE Flag Cnt"]  # 0
            if config["Down/Up Ratio"]:
                self.normalize_through_rows("Down/Up Ratio") #todo test
            if config["Pkt Size Avg"]:
                self.half_sigmoid("Pkt Size Avg")
            if config["Fwd Seg Size Avg"]:
                self.half_sigmoid("Fwd Seg Size Avg")
            if config["Bwd Seg Size Avg"]:
                self.half_sigmoid("Bwd Seg Size Avg")
            if config["Fwd Byts/b Avg"]:  # 0
                self.new_df["Fwd Byts/b Avg"] = self.old_df["Fwd Byts/b Avg"]
            if config["Fwd Pkts/b Avg"]:  # 0
                self.new_df["Fwd Pkts/b Avg"] = self.old_df["Fwd Pkts/b Avg"]
            if config["Fwd Blk Rate Avg"]:  # 0
                self.new_df["Fwd Blk Rate Avg"] = self.old_df["Fwd Blk Rate Avg"]
            if config["Bwd Byts/b Avg"]:  # 0
                self.new_df["Bwd Byts/b Avg"] = self.old_df["Bwd Byts/b Avg"]
            if config["Bwd Pkts/b Avg"]:  # 0
                self.new_df["Bwd Pkts/b Avg"] = self.old_df["Bwd Pkts/b Avg"]
            if config["Bwd Blk Rate Avg"]:  # 0
                self.new_df["Bwd Blk Rate Avg"] = self.old_df["Bwd Blk Rate Avg"]
            if config["Subflow Fwd Pkts"]:
                self.half_sigmoid("Subflow Fwd Pkts")
            if config["Subflow Fwd Byts"]:
                self.half_sigmoid("Subflow Fwd Byts")
            if config["Subflow Bwd Pkts"]:
                self.half_sigmoid("Subflow Bwd Pkts")
            if config["Subflow Bwd Byts"]:
                self.half_sigmoid("Subflow Bwd Byts")
            if config["Init Fwd Win Byts"]:  # -1
                self.new_df["Init Fwd Win Byts"] = self.old_df["Init Fwd Win Byts"]
            if config["Init Bwd Win Byts"]:
                self.divide_by_value("Init Bwd Win Byts", 65535)
            if config["Fwd Act Data Pkts"]:
                self.half_sigmoid("Fwd Act Data Pkts")
            if config["Fwd Seg Size Min"]:  # 0
                self.new_df["Fwd Seg Size Min"] = self.old_df["Fwd Seg Size Min"]
            if config["Active Mean"]:
                self.half_sigmoid("Active Mean")
            if config["Active Std"]:
                self.half_sigmoid("Active Std")
            if config["Active Max"]:
                self.half_sigmoid("Active Max")
            if config["Active Min"]:
                self.half_sigmoid("Active Min")
            if config["Idle Mean"]:
                self.half_sigmoid("Idle Mean")
            if config["Idle Std"]:
                self.half_sigmoid("Idle Std")
            if config["Idle Max"]:
                self.half_sigmoid("Idle Max")
            if config["Idle Min"]:
                self.half_sigmoid("Idle Min")
            if config["Label"]:
                self.process_label("Label")
        if dataset == "KDD":
            config = config["KDD"]
            if config["duration"]:
                self.half_sigmoid("duration")
            if config["protocol_type"]:
                self.process_one_hot("protocol_type")
            if config["service"]:
                self.process_one_hot("service")
            if config["flag"]:
                self.process_one_hot("flag")
            if config["src_bytes"]:
                self.half_sigmoid("src_bytes")
            if config["dst_bytes"]:
                self.half_sigmoid("dst_bytes")
            if config["land"]:
                self.new_df["land"] = self.old_df["land"]
            if config["wrong_fragment"]:
                self.half_sigmoid("wrong_fragment")
            if config["urgent"]:
                self.half_sigmoid("urgent")
            if config["hot"]:
                self.half_sigmoid("hot")
            if config["num_failed_logins"]:
                self.half_sigmoid("num_failed_logins")
            if config["logged_in"]:
                self.new_df["logged_in"] = self.old_df["logged_in"]
            if config["num_compromised"]:
                self.half_sigmoid("num_compromised")
            if config["root_shell"]:
                self.half_sigmoid("root_shell")
            if config["su_attempted"]:
                self.half_sigmoid("su_attempted")
            if config["num_root"]:
                self.half_sigmoid("num_root")
            if config["num_file_creations"]:
                self.half_sigmoid("num_file_creations")
            if config["num_shells"]:
                self.half_sigmoid("num_shells")
            if config["num_access_files"]:
                self.half_sigmoid("num_access_files")
            if config["num_outbound_cmds"]:
                self.half_sigmoid("num_outbound_cmds")
            if config["is_host_login"]:
                self.new_df["is_host_login"] = self.old_df["is_host_login"]
            if config["is_guest_login"]:
                self.new_df["is_guest_login"] = self.old_df["is_guest_login"]
            if config["count"]:
                self.divide_by_value("count",511)
            if config["srv_count"]:
                self.divide_by_value("srv_count",511)
            if config["serror_rate"]:
                self.new_df["serror_rate"] = self.old_df["serror_rate"]
            if config["srv_serror_rate"]:
                self.new_df["srv_serror_rate"] = self.old_df["srv_serror_rate"]
            if config["rerror_rate"]:
                self.new_df["rerror_rate"] = self.old_df["rerror_rate"]
            if config["srv_rerror_rate"]:
                self.new_df["srv_rerror_rate"] = self.old_df["srv_rerror_rate"]
            if config["same_srv_rate"]:
                self.new_df["same_srv_rate"] = self.old_df["same_srv_rate"]
            if config["diff_srv_rate"]:
                self.new_df["diff_srv_rate"] = self.old_df["diff_srv_rate"]
            if config["srv_diff_host_rate"]:
                self.new_df["srv_diff_host_rate"] = self.old_df["srv_diff_host_rate"]
            if config["dst_host_count"]:
                self.divide_by_value("dst_host_count",255)
            if config["dst_host_srv_count"]:
                self.divide_by_value("dst_host_srv_count",255)
            if config["dst_host_same_srv_rate"]:
                self.new_df["dst_host_same_srv_rate"] = self.old_df["dst_host_same_srv_rate"]
            if config["dst_host_diff_srv_rate"]:
                self.new_df["dst_host_diff_srv_rate"] = self.old_df["dst_host_diff_srv_rate"]
            if config["dst_host_same_src_port_rate"]:
                self.new_df["dst_host_same_src_port_rate"] = self.old_df["dst_host_same_src_port_rate"]
            if config["dst_host_srv_diff_host_rate"]:
                self.new_df["dst_host_srv_diff_host_rate"] = self.old_df["dst_host_srv_diff_host_rate"]
            if config["dst_host_serror_rate"]:
                self.new_df["dst_host_serror_rate"] = self.old_df["dst_host_serror_rate"]
            if config["dst_host_srv_serror_rate"]:
                self.new_df["dst_host_srv_serror_rate"] = self.old_df["dst_host_srv_serror_rate"]
            if config["dst_host_rerror_rate"]:
                self.new_df["dst_host_rerror_rate"] = self.old_df["dst_host_rerror_rate"]
            if config["dst_host_srv_rerror_rate"]:
                self.new_df["dst_host_srv_rerror_rate"] = self.old_df["dst_host_srv_rerror_rate"]
            if config["label"]:
                self.process_label("label")
        if dataset == "Przemek":
            config = config["Przemek"]
            if config["Type"]:
                self.test("Type")
            if config["TimeReceived"]:
                self.new_df["TimeReceived"] = self.old_df["TimeReceived"]
            if config["SequenceNum"]:
                self.new_df["SequenceNum"] = self.old_df["SequenceNum"]
            if config["SamplingRate"]:
                self.test("SamplingRate")
            if config["SamplerAddress"]:
                self.test("SamplerAddress")
            if config["TimeFlowStart"]:
                self.new_df["TimeFlowStart"] = self.old_df["TimeFlowStart"]
                #self.new_df["Duration"] = self.old_df["TimeFlowEnd"] - self.old_df["TimeFlowStart"]
                #uni = self.new_df["Duration"].unique()
                #print(len(uni))
                #uni.sort()
                #print(uni)
                #self.process_one_hot("Protocol")
            if config["TimeFlowEnd"]:
                self.new_df["TimeFlowEnd"] = self.old_df["TimeFlowEnd"]
                #self.process_one_hot("Protocol")
            if config["Bytes"]:
                self.normalize_through_rows("Bytes")
            if config["Packets"]:
                self.test("Packets")
            if config["SrcAddr"]:
                self.process_ip("SrcAddr")
            if config["DstAddr"]:
                self.process_ip("DstAddr")
            if config["Etype"]:
                self.process_one_hot("Etype")
            if config["Proto"]:
                self.process_one_hot("Proto")
            if config["SrcPort"]:
                self.divide_by_value("SrcPort", 65535)
            if config["DstPort"]:
                self.divide_by_value("DstPort", 65535)
            self.new_df["Label"] = self.old_df["Label"]
        self.new_df = self.new_df.fillna(0)
        print(self.new_df)
        print(self.new_df.head)
        return(self.new_df)




