from pathlib import Path
import pandas as pd
from FeaturesConversion_new import FeaturesConverter
import yaml



# processing danych z datasetu
converter = FeaturesConverter("/dataset/")
# config wyboru, które dane zawrzeć
config_path = Path("data_config.yaml")
config = yaml.load(config_path.open(mode="r"), Loader=yaml.FullLoader)
dataset = "Przemek"
processed_data = converter.process_data(config, dataset)

row_num = len(processed_data)
#processed_data = processed_data.sample(frac = 1)


train_data = processed_data.iloc[:]
normal_train = train_data[train_data["Label"] == 0]
print("Normal train:"+ str(len(normal_train)))
anormal_train = train_data[train_data["Label"] != 0]
print("Anormal train:"+ str(len(anormal_train)))
normal_train.to_csv("out_preprocess/normaltrain.csv", index=False)
anormal_train.to_csv("out_preprocess/anormaltrain.csv", index=False)

#test_data = processed_data.iloc[:int(0.2*row_num)]
#normal_test = test_data[test_data["Label"] == 0]
#print("Normal test:"+ str(len(normal_test)))
#anormal_test = test_data[test_data["Label"] != 0]
#print("Anormal test:"+ str(len(anormal_test)))
#normal_test.to_csv("out_preprocess/normaltest.csv", index=False)
#anormal_test.to_csv("out_preprocess/anormaltest.csv", index=False)
