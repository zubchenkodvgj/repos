
import sys
import pandas as pd
import numpy as np
import pyodbc
import joblib
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QMessageBox
)

class XGBPredictorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('XGBoost Прогноз из СУБД')
        self.resize(800, 600)
        self.model = joblib.load('xgb_model.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.features_to_scale = ['dow', 'n_year_week', 'n_month_week', 'n_week_week', 'prih_unt_int', 'prih_rub_int', 'mkdn_unt_int', 'mkdn_rub_int', 'discount_rub_int', 'apoh_int', 'eoh_unt_int', 'eoh_rub_int', 'discount_rub_prc', 'mkdn_rub_int_5days', 'sms', 'st_area', 'napoln', 'napoln_rub', 'cr', 'sls_rub_int_year', 'visiors_year', 'sls_rub_online', 'sls_unt_online', 'sls_rub_online_prc', 'sls_unt_online_prc']
        self.features_to_pred = ['n_week_week_sin', 'n_week_week_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'dow', 'prih_unt_int', 'prih_rub_int', 'mkdn_unt_int', 'mkdn_rub_int', 'discount_rub_int', 'apoh_int', 'eoh_unt_int', 'eoh_rub_int', 'discount_rub_prc', 'mkdn_rub_int_5days', 'sms', 'st_area', 'cr', 'sls_rub_int_year', 'visiors_year', 'sls_rub_online', 'sls_unt_online', 'sls_rub_online_prc', 'sls_unt_online_prc']
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText('Введите SQL-запрос...')
        layout.addWidget(QLabel('SQL-запрос к базе данных:'))
        layout.addWidget(self.query_input)

        self.run_button = QPushButton('Сделать прогноз')
        self.run_button.clicked.connect(self.run_prediction)
        layout.addWidget(self.run_button)

        self.table = QTableWidget()
        layout.addWidget(self.table)

        self.save_button = QPushButton('Сохранить результат в CSV')
        self.save_button.clicked.connect(self.save_csv)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def run_prediction(self):
        try:
            conn = pyodbc.connect("DSN=vertica")
            query = self.query_input.toPlainText()
            df = pd.read_sql(query, conn)
            df = df.select_dtypes(exclude=['object'])
            df_scaled = df.copy()
            df_scaled[self.features_to_scale] = self.scaler.transform(df_scaled[self.features_to_scale])
            df_scaled['month_sin'] = np.sin(2 * np.pi * df_scaled['n_month_week'] / 12)
            df_scaled['month_cos'] = np.cos(2 * np.pi * df_scaled['n_month_week'] / 12)
            df_scaled['dow_sin'] = np.sin(2 * np.pi * df_scaled['dow'] / 7)
            df_scaled['dow_cos'] = np.cos(2 * np.pi * df_scaled['dow'] / 7)
            df_scaled['n_week_week_sin'] = np.sin(2 * np.pi * df_scaled['n_week_week'] / 52)
            df_scaled['n_week_week_cos'] = np.cos(2 * np.pi * df_scaled['n_week_week'] / 52)
            X = df_scaled[self.features_to_pred]
            preds = np.expm1(self.model.predict(X))
            df_result = df.copy()
            df_result['prediction'] = preds
            self.last_result = df_result
            self.show_table(df_result)
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', str(e))

    def show_table(self, df):
        self.table.setColumnCount(len(df.columns))
        self.table.setRowCount(len(df))
        self.table.setHorizontalHeaderLabels(df.columns.astype(str))
        for i in range(len(df)):
            for j in range(len(df.columns)):
                self.table.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))

    def save_csv(self):
        if hasattr(self, 'last_result'):
            path, _ = QFileDialog.getSaveFileName(self, 'Сохранить CSV', '', 'CSV Files (*.csv)')
            if path:
                self.last_result.to_csv(path, index=False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = XGBPredictorGUI()
    gui.show()
    sys.exit(app.exec_())
