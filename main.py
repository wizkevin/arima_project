import data_preparation, arima_model

data_preparation_object = data_preparation.DataPreparation(csv_path="monthly-sunspots.csv", ratio=(0.7, 0.2))
data_preparation_object.show_dataset()
data_preparation_object.verify_tendance_or_saisonalite()

t_train, t_validation, t_test, y_train, y_validation, y_test, train_dataset_df, validation_dataset_df, test_dataset_df, number_of_sunspots_df = data_preparation_object.prepare_data_for_arima_model()
arima_model_object = arima_model.ArimaModel(t_train, t_validation, t_test, y_train, y_validation, y_test, train_dataset_df,validation_dataset_df, test_dataset_df, number_of_sunspots_df)
arima_model_object.training_arima(p=33, d=0, q=0)
arima_model_object.show_forecast_of_arima_model()