import pmdarima as pm
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import statsmodels.graphics.tsaplots as stg
from tqdm import tqdm

class ArimaModel:
    def __init__(self, t_train, t_validation, t_test, y_train, y_validation, y_test, train_dataset_df, validation_dataset_df, test_dataset_df, number_of_sunspots_df):
        """
        Initialise une instance de la classe ArimaModel avec les paramètres nécessaires pour l'entraînement et la prédiction ARIMA.
        
        Args:
            p (int): Ordre du terme AR (AutoRégressif).
            d (int): Ordre de la différenciation.
            q (int): Ordre du terme MA (Moyenne Mobile).
            t_train (Index time): Index time de l'ensemble d'entraînement.
            t_validation (Index time): Index time de l'ensemble de validation.
            t_test (Index time): Index time de l'ensemble de test.
            y_train (Sunspots): Valeurs du nombre de taches solaires de l'ensemble d'entraînement.
            y_validation (Sunspots): Valeurs du nombre de taches solaires de l'ensemble de validation.
            y_test (Sunspots): Valeurs du nombre de taches solaires de l'ensemble de test.
        """
        self.t_train = t_train
        self.t_validation = t_validation
        self.t_test = t_test
        
        self.y_train = y_train
        self.y_validation = y_validation
        self.y_test = y_test
        
        self.test_dataset_df = test_dataset_df
        self.train_dataset_df = train_dataset_df
        self.validation_dataset_df = validation_dataset_df
        self.number_of_sunspots_df = number_of_sunspots_df

    def training_arima(self, p, d, q):
        """
        Entraîne le modèle ARIMA en utilisant les données d'entraînement.
        Affiche le résumé du modèle et enregistre le résumé dans un fichier 'summary.txt'.
        """
        self.p = p
        self.d = d
        self.q = q
        
        self.arima = ARIMA(self.y_train, order=(self.p, self.d, self.q))
       # J'utilise range(len(self.y_train)) dans la boucle for pour obtenir les indices de chaque itération
        # with tqdm(total=len(self.y_train)) as pbar:
        #     for i in range(len(self.y_train)):
                # self.arima_model_fit = self.arima.fit()
                # Mise à jour la barre de progression à chaque itération
                # pbar.update(1)
        self.arima_model_fit = self.arima.fit()
        print(self.arima_model_fit.summary())

    def show_forecast_of_arima_model(self):
        """
        Affiche les prédictions du modèle ARIMA pour les ensembles de train, validation et test.
        Calcule le RMSE (Root Mean Squared Error) pour les prédictions du test.
        """
        self.train_dataset_df = self.train_dataset_df.drop('Month',axis=1)
        
        # Make time series predictions
        self.residuals = self.arima_model_fit.resid[1:]
        # Convertir self.residuals en Series
        residuals_series = pd.Series(self.residuals)
        
        # Affichage des résidus et de leur densité
        fig, ax = plt.subplots(1,2)
        residuals_series.plot(title='Residuals', ax = ax[0])
        residuals_series.plot(title='Density', kind='kde', ax=ax[1])
        plt.show()
        
        acf_residual = stg.plot_acf(residuals_series)
        plt.show()
        
        pacf_residual = stg.plot_pacf(residuals_series)
        plt.show()
        
        self.forecast_test = self.arima_model_fit.forecast(len(self.test_dataset_df))
        
        # Ajouter des valeurs manquantes pour correspondre à la longueur de l'index
        # forecast_values = [None] * len(self.train_dataset_df) + list(self.forecast_test)
        forecast_values = [None] * (len(self.number_of_sunspots_df) - len(self.forecast_test)) + list(self.forecast_test)

        # Créer une nouvelle colonne avec les valeurs ajustées
        self.number_of_sunspots_df["forecast_manuel"] = forecast_values
        self.number_of_sunspots_df = self.number_of_sunspots_df.drop('index_time',axis=1)
        
        self.number_of_sunspots_df.plot()
        
        
        # Réorganiser vos données pour obtenir le tableau unidimensionnel
        y_train = np.array(self.train_dataset_df).flatten()

        # Utiliser le tableau unidimensionnel pour la fonction auto_arima
        auto_arima = pm.auto_arima(y_train, stepwise=False, seasonal=False)
        print(auto_arima)
        
        print(auto_arima.summary())
        
        # forecast_test_auto = auto_arima.predict(n_periods=len(self.test_dataset_df))
        # # Créer une série avec les valeurs ajustées et l'index approprié
        # forecast_series = [None]* len(self.train_dataset_df + self.validation_dataset_df) + list(forecast_test_auto)

        # # Assigner la série à la colonne 'forecast_auto'
        # self.number_of_sunspots_df['forecast_auto'] = forecast_series
        
        # self.number_of_sunspots_df.plot()
        
        train_predictions = self.arima_model_fit.predict(
            start=0, end=len(self.t_train) + len(self.t_validation) + len(self.t_test) - 1, typ='levels')
        validation_predictions = self.arima_model_fit.predict(
            start=len(self.t_train), end=len(self.t_train) + len(self.t_validation) - 1, typ='levels')
        test_predictions = self.arima_model_fit.predict(
            start=len(self.t_train) + len(self.t_validation), end=len(self.number_of_sunspots_df) - 1, typ='levels')
        
        train_total = self.arima_model_fit.predict(
            start=0, end=len(self.number_of_sunspots_df) - 1, typ='levels')
        print(test_predictions)
        rmse = np.sqrt(mean_squared_error(self.y_test, test_predictions))

        print("Début mesure RMSE".center(50, "-"))
        print("RMSE:", rmse)
        print("Fin mesure RMSE".center(50, "-"))

        plt.figure(figsize=(15, 6))
        # train base
        plt.plot(self.t_train, self.y_train, "o", color='b', label="Train base")
        # validation base
        plt.plot(self.t_validation, self.y_validation, "o", color='r', label="Validation base")
        # test base
        plt.plot(self.t_test, self.y_test, "o", color='g', label="Test base")

        plt.plot(np.concatenate([self.t_train, self.t_validation, self.t_test]), train_predictions, "-", color='b', label="Train prediction")
        plt.plot(self.t_validation, validation_predictions, "-", color='r', label="Validation prediction")
        plt.plot(self.t_test, test_predictions, "-", color='g', label="Test prediction")

        plt.legend()
        plt.show()


        