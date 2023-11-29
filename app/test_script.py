import json
import logging

import pandas as pd

# import torch
# import torch.nn.functional as F
# from models import VAE

logger = logging.getLogger(__name__)

'''
input:
{
    "type": "SFLOW_5",
    "time_received_ns": 1681583295157626000,
    "sequence_num": 2999,
    "sampling_rate": 100,
    "sampler_address": "192.168.0.1",
    "time_flow_start_ns": 1681583295157626000,
    "time_flow_end_ns": 1681583295157626000,
    "bytes": 1500,
    "packets": 1,
    "src_addr": "fd01::1",
    "dst_addr": "fd01::2",
    "etype": "IPv6",
    "proto": "TCP",
    "src_port": 443,
    "dst_port": 50001
}
'''


#
# def process_ip(ip):
#     if isinstance(ip, str):
#         octets = ip.split('.')
#         return [float(octet) / 255 for octet in octets]
#     else:
#         # If the input is not a string, return an array of zeros
#         return [0.0, 0.0, 0.0, 0.0]
#
# def one_hot_encode(input, values):
#     # Definiowanie możliwych wartości dla 'proto'
#     return [1 if input == value else 0 for value in values]
#
# def normalize_column(tensor, column_index):
#     """Normalizuje wybraną kolumnę tensora przy użyciu normalizacji min-max."""
#     min_val = torch.min(tensor[:, column_index])
#     max_val = torch.max(tensor[:, column_index])
#     if max_val - min_val != 0:
#         tensor[:, column_index] = (tensor[:, column_index] - min_val) / (max_val - min_val)
#     else:
#         # Możesz zdecydować co zrobić, gdy min i max są takie same
#         # Na przykład, ustaw wszystkie wartości na 0 lub zachowaj je bez zmian
#         tensor[:, column_index] = 0  # lub zachowaj bez zmian
#
#     return tensor

def load_dict(data):
    # load csv
    dataframes = [pd.DataFrame(json.loads(i)) for i in data]
    df = pd.concat(dataframes, ignore_index=True)
    # create df with only necesaries
    new_df = pd.DataFrame()
    new_df["type"] = df["Type"]
    new_df["time_received_ns"] = df["TimeReceived"]
    new_df["sequence_num"] = df["SequenceNum"]
    new_df["sampling_rate"] = df["SamplingRate"]
    new_df["sampler_address"] = df["SamplerAddress"]
    new_df["time_flow_start_ns"] = df["TimeFlowStart"]
    new_df["time_flow_end_ns"] = df["TimeFlowEnd"]
    new_df["bytes"] = df["Bytes"]
    new_df["packets"] = df["Packets"]
    new_df["src_addr"] = df["SrcAddr"]
    new_df["dst_addr"] = df["DstAddr"]
    new_df["etype"] = df["Etype"]
    new_df["proto"] = df["Proto"]
    new_df["src_port"] = df["SrcPort"]
    new_df["dst_port"] = df["DstPort"]
    list_of_dicts = [row.to_dict() for _, row in new_df.iterrows()]

    return list_of_dicts


# class Tester:
#     def __init__(self,list_of_dicts):
#         self.buffer_size  = 5
#         self.threshold = 0.042 # Threshold obliczony na zbiiorze danych treningowych
#         self.etype_values = [2048, 2054]
#         self.proto_values = [0, 1, 6]
#         self.list_of_dicts = list_of_dicts
#
#         self.encoder = VAE.Encoder(100, 64, 2, 3, 16)
#         self.decoder = VAE.Decoder(100, 64, 2, 3, 16)
#
#         self.encoder.load_state_dict(torch.load("saved_models_Test/Encoder.pth", map_location=torch.device('cpu')), strict=True)
#         self.decoder.load_state_dict(torch.load("saved_models_Test/Decoder.pth", map_location=torch.device('cpu')), strict=True)
#
#         self.encoder.eval()
#         self.decoder.eval()
#         self.normal_count = 0
#         self.anomaly_count = 0
#
#         #encoder.to(device)
#         #decoder.to(device)
#
#
#
#     def process_dict(self):
#         # to co tu sie dzieje to jest wymuszone ograniczonym zbiorem danych
#         # do sieci potrzebne jest 5 próbek
#         '''
#           Type: False
#           TimeReceived: True
#           SequenceNum: True
#           SamplingRate: False
#           SamplerAddress: False
#           TimeFlowStart: True
#           TimeFlowEnd: True
#           Bytes: True
#           Packets: False
#           SrcAddr: True
#           DstAddr: True
#           Etype: True
#           Proto: True
#           SrcPort: True
#           DstPort: True'''
#         for i in range(len(self.list_of_dicts) - self.buffer_size + 1):
#             processed_data = []
#             sequence = self.list_of_dicts[i:i + self.buffer_size]  # cos takiego przewiduje ze bedziesz podawal
#             scr_addr = sequence[0]["src_addr"]
#             dst_addr = sequence[0]["dst_addr"]
#             src_port = sequence[0]["src_port"]
#             dst_port = sequence[0]["dst_port"]
#
#             for flow in sequence:
#                 processed_item = []
#                 processed_item.append(flow["time_received_ns"])
#                 processed_item.append(flow["sequence_num"])
#                 processed_item.append(flow["time_flow_start_ns"])
#                 processed_item.append(flow["time_flow_end_ns"])
#                 # min i max wziety z dataset
#                 processed_item.append((flow["bytes"] - 64) / (238 - 64))
#                 processed_item.extend(process_ip(flow["src_addr"]))
#                 processed_item.extend(process_ip(flow["dst_addr"]))
#                 processed_item.extend(one_hot_encode(flow["etype"], self.etype_values))
#                 processed_item.extend(one_hot_encode(flow["proto"], self.proto_values))
#                 processed_item.append(flow["src_port"] / 65535)
#                 processed_item.append(flow["dst_port"] / 65535)
#                 processed_data.append(processed_item)
#
#
#             tensor = torch.tensor(processed_data, dtype=torch.float32)
#
#
#             tensor = normalize_column(tensor, 0)
#             tensor = normalize_column(tensor, 1)
#             tensor = normalize_column(tensor, 2)
#             tensor = normalize_column(tensor, 3)
#
#             #print(tensor)
#
#             #b, c, g = tensor.shape
#             tensor = tensor.reshape(1, -1)
#             #tensor = tensor[:, :, :-1].reshape(b, -1).to(device)
#             #print(tensor)
#
#             mu, logvar, z = self.encoder(tensor)
#             reconstructed = self.decoder(z)
#
#             recons_loss = F.mse_loss(reconstructed, tensor, reduction="none")
#             predict = recons_loss.mean(-1)
#             if predict < self.threshold:
#                 self.normal_count += 1
#                 print(f"normal {predict} src_addr: {scr_addr} dst_addr: {dst_addr} src_port: {src_port} dst_port: {dst_port}")
#             else:
#                 self.anomaly_count += 1
#                 print(f"anormal {predict} src_addr: {scr_addr} dst_addr: {dst_addr} src_port: {src_port} dst_port: {dst_port}")
#
#         print("Normal:")
#         print(self.normal_count)
#         print("Anormal:")
#         print(self.anomaly_count)


def process(data):
    list_of_dicts = load_dict(data)
    logger.warning(f"Received message: {list_of_dicts}")
    if len(list_of_dicts) >= 5:
        # send post request to security controller
        return list_of_dicts
