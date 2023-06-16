import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.graphics.tsaplots as stg
from statsmodels.tsa.stattools import adfuller

"""
    Ce jeu de données décrit un décompte mensuel du nombre de taches solaires observées pendant 
    un peu plus de 230 ans (1749-1983).
    Les unités sont un comptage et il y a 2.820 observations.

Returns:
    _type_: _description_
"""

class DataPreparation:
    
    """Cette classe me permet de gérer le jeu de données"""
    def __init__(self, csv_path, ratio):

        self.number_of_sunspots_df = pd.read_csv(csv_path, sep=",")
        self.dataset_length = len(self.number_of_sunspots_df)

        # Calcul des indices de division du jeu de données en fonction des ratios spécifiés
        self.ratio = ratio
        self.index_split_1 = int(self.dataset_length * self.ratio[0])
        self.index_split_2 = int(self.dataset_length * (self.ratio[0]+ self.ratio[1]))
        
    def prepare_data_for_arima_model(self):
        # normalise mon jeu données tout en sauvegardant le coefficient de normalisation
        # coefficient_normalisation = self.number_of_sunspots_df["Sunspots"].values.max()
        # self.number_of_sunspots_df["Sunspots"] /= coefficient_normalisation
        
        self.number_of_sunspots_df.info()

        # j'encode ma date sous forme d'entier
        self.number_of_sunspots_df["index_time"] = np.array([index_date for index_date in range(0, self.dataset_length)])
        
        # J'effectue le split train / validation  /test
        # Séparer les données en ensembles d'entraînement, de validation et de test, 
        train_dataset_df = self.number_of_sunspots_df.loc[ : self.index_split_1-1]
        validation_dataset_df = self.number_of_sunspots_df.loc[self.index_split_1 : self.index_split_2-1]
        test_dataset_df = self.number_of_sunspots_df.loc[self.index_split_2 : ]


        # Extraction des valeurs de la colonne index_time des dataframes de train / validation / test
        t_train = train_dataset_df["index_time"].values
        t_validation = validation_dataset_df["index_time"].values
        t_test = test_dataset_df["index_time"].values


        # Extraction des valeurs de la colonne Sunspots des dataframes de train / validation / test
        y_train = train_dataset_df["Sunspots"].values
        y_validation = validation_dataset_df["Sunspots"].values
        y_test = test_dataset_df["Sunspots"].values
        
        number_of_sunspots_df = self.number_of_sunspots_df
        
        return t_train, t_validation, t_test, y_train, y_validation, y_test, train_dataset_df, test_dataset_df, number_of_sunspots_df
  
    def show_dataset(self):
        """
        Etape 0
        Permet d'afficher le contenu de mon fichier csv sous la forme d'un graphique
        """
        self.prepare_data_for_arima_model()
        
        plt.figure(figsize=(15, 6))
        plt.plot(self.number_of_sunspots_df["index_time"], self.number_of_sunspots_df["Sunspots"])
        plt.xlabel("Index Time")
        plt.ylabel("Sunspots")
        plt.title("Time Series of Sunspots")
        # sns.scatterplot(x="index_time", y= "Sunspots", data=self.number_of_sunspots_df)
        plt.show()


        return None
    
    def verify_tendance_or_saisonalite(self):
        self.prepare_data_for_arima_model()

        # Décomposition de la série temporelle en tendance, saisonnalité et résidus
        decomposition = seasonal_decompose(self.number_of_sunspots_df["Sunspots"], period=12)  # Période saisonnière à ajuster selon votre série temporelle
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residuals = decomposition.resid

        # Plot de la tendance de la série temporelle
        plt.figure(figsize=(12, 6))
        plt.plot(self.number_of_sunspots_df["index_time"], trend)
        plt.title("Tendance de la série temporelle")
        plt.xlabel("Temps")
        plt.ylabel("Tendance")
        plt.show()

        # Plot de la composante saisonnière de la série temporelle
        plt.figure(figsize=(12, 6))
        plt.plot(self.number_of_sunspots_df["index_time"], seasonal)
        plt.title("Composante saisonnière de la série temporelle")
        plt.xlabel("Temps")
        plt.ylabel("Composante saisonnière")
        plt.show()

        # Plot des résidus de la décomposition
        plt.figure(figsize=(12, 6))
        plt.plot(self.number_of_sunspots_df["index_time"], residuals)
        plt.title("Résidus de la décomposition de la série temporelle")
        plt.xlabel("Temps")
        plt.ylabel("Résidus")
        plt.show()

        # Trouver p
        # Calcul des autocorrélations partielles
        self.df1 = self.number_of_sunspots_df.copy()
        
        # j'encode ma date sous forme d'entier
        # self.df1["index_time"] = np.array([index_date for index_date in range(0, self.dataset_length)])
        self.df1["index_time"] = pd.to_datetime(self.df1.Month)
        self.df1 = self.df1.drop('Month',axis=1)
        self.df1 = self.df1.drop('index_time',axis=1)
        
        result_adf1 = adfuller(self.df1)
        print(f"ADF= {result_adf1}")
        print(f"p-value= {result_adf1[1]}")
        
        self.df1_diff = self.df1.diff().dropna()
        self.df1_diff.plot()
        
        # Autocorrelation
        stg.plot_acf(self.df1)
        plt.show()
        
        # Partial autocorrelation
        stg.plot_pacf(self.df1)
        plt.show()
    
        adf_test = adfuller(self.df1_diff)
        print(f"ADF= {adf_test}")
        print(f"p-value= {adf_test[1]}")
