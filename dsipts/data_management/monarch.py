import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import pickle 
import os
import shutil
from datetime import datetime
from distutils.util import strtobool
from typing import Union


# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
)-> pd.DataFrame:
    """I copied this function from the repo

    Args:
        full_file_path_and_name (str): path
        replace_missing_vals_with (str, optional): replace not valid numbers. Defaults to "NaN".
        value_column_name (str, optional):. Defaults to "series_value". 


    Returns:
        pd.DataFrame: output data frame
    """
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )
        
        
def get_freq(freq)->str:
    """Get the frequency based on the string reported. I don't think there are all the possibilities here

    Args:
        freq (str): string coming from 

    Returns:
        str: pandas frequency format
    """
    if freq =='10_minutes':
        return '600s'
    elif freq == 'hourly':
        return 'H'
    else:
        return 'D'     
        
        
class Monarch():
    
    def __init__(self,filename:str,baseUrl:str ='https://forecastingdata.org/', rebuild:bool =False):
        """Class for downloading datasets listed here https://forecastingdata.org/ 

        Args:
            filename (str):  name of the class, used for saving
            baseUrl (str, optional): url to the source page. Defaults to 'https://forecastingdata.org/'.
            rebuild (bool, optional):  if true the table will be loaded from the webpage otherwise it will be loaded from the saved file. Defaults to False.
        """
        self.baseUrl = baseUrl
        self.downloaded = {}
        if rebuild==False:
            print(filename)
            if os.path.exists(filename+'.pkl'):
                self.load(filename)
            else:
                self.get_table(baseUrl)
                self.save(filename)
        else:
            self.get_table(baseUrl)
            self.save(filename)

    def get_table(self, baseUrl):
        """    
        Used in the init
        :meta private:
        """
        with requests.Session() as s:
            r = s.get(baseUrl)
        soup = bs(r.content)
        header = []
        for x in soup.find("table", {"class": "responsive-table sortable"}).find('thead').find_all('th'):
            header.append(x.text)
        header

        tot= []
        for row in soup.find("table", {"class": "responsive-table sortable"}).find('tbody').find_all('tr'):
            row_data = []
            links = {}
            for i,column in enumerate(row.find_all('td')):
                tmp_links = column.find_all('a')
                if len(tmp_links)>0:

                    for x in tmp_links:
                        if 'zenodo' in x['href']:
                            links[x.text] = x['href']
                            i_links = i
                row_data.append(column.text)
            for dataset in links:
                row_to_insert = {}
                for j, head in enumerate(header):
                    if j!=i_links:
                        row_to_insert[header[j]] = row_data[j]
                    else:
                        row_to_insert['freq'] = dataset
                        row_to_insert[header[j]] = links[dataset]
                tot.append(row_to_insert)

        tot = pd.DataFrame(tot)
        tot['id']  = tot.Download.apply(lambda x:int(x.split('/')[-1]))
        self.table = tot.copy()
   
    def save(self, filename:str)-> None:
        """Save the monarch structure

        Args:
            filename (str): name of the file to generate
        """
        print('Saving')
        with open(f'{filename}.pkl','wb') as f:
            params =  self.__dict__.copy()
            #for k in ['data','data_train','data_test','data_validation']:
            #    if k in params.keys():
            #        _ = params.pop(k)
            pickle.dump(params,f)
    def load(self, filename:str)-> None:
        """Load a monarch structure

        Args:
            filename (str): filename to load
        """
        print('Loading')
        with open(filename+'.pkl','rb') as f:
            params = pickle.load(f)
            for p in params:
                setattr(self,p, params[p])    
   
    def download_dataset(self,path: str,id:int ,rebuild=False)->None:
        """download a specific dataset 

        Args:
            path (str): path in which save the data
            id (int): id of the dataset
            rebuild (bool, optional): if true the dataset will be re-downloaded. Defaults to False.
        """
            
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)
        if os.path.exists(os.path.join(path,str(id))):
            if rebuild:
                file = self._download(url = self.table.Download[self.table.id== id].values[0], path = os.path.join(path,str(id)))
                self.downloaded[id] = f'{path}/{id}/{file}'
            else:
                pass
        else:
            file = self._download(url = self.table.Download[self.table.id== id].values[0] , path = os.path.join(path,str(id)))
            self.downloaded[id] = f'{path}/{id}/{file}'
            
    def _download(self,url, path)->str:
        """ get data
        :meta private:
        """
        with requests.Session() as s:
            r = s.get(url)
            soup = bs(r.content)

            url = soup.find("link", {"type": "application/zip"})['href']
            print(url)
            with open(path+'.zip', "wb") as f:
                f.write(s.get(url).content)
        
        shutil.unpack_archive(path+'.zip', path)
        os.remove(path+'.zip')
        return os.listdir(path)[0]
       
    def generate_dataset(self, id:int)-> Union[None, pd.DataFrame]:
        """Parse the id-th dataset in a convient format and return a pandas dataset

        Args:
            id (int): id of the dataset

        Returns:
            None or pd.DataFrame: dataframe
        """

        if id not in self.downloaded.keys():
            print('please call first download dataset')
            return None
        else:
            return convert_tsf_to_dataframe(self.downloaded[id])