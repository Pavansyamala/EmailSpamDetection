import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('D:/EmailSpamClassification/notebook/spam_ham_dataset.csv')
            df.drop(columns=['Unnamed: 0'] , inplace = True)
            df.drop(columns = ['label_num'] , inplace = True)
            ROS = RandomOverSampler()
            text_resampled , label_num_resampled = ROS.fit_resample(df['text'].values.reshape(-1,1) , df['label'].values.reshape(-1,1))
            resampled_data = pd.DataFrame(label_num_resampled ).reset_index().merge(how = 'inner' , on = 'index', right = pd.DataFrame(text_resampled).reset_index()).drop(columns = ['index'])
            resampled_data['label'] = resampled_data['0_x']
            resampled_data['text'] = resampled_data['0_y']      
            resampled_data.drop(columns = ['0_x','0_y'] , inplace = True)
            resampled_data['text'] = resampled_data['text'].apply(lambda x : x.split('Subject: ')[-1])
            resampled_data['Subject'] = resampled_data['text']
            resampled_data.drop(columns=['text'] , inplace = True)
            print(resampled_data.head())
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            resampled_data.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(resampled_data,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
