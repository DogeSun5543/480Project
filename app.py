import streamlit as st
import pandas as pd
import pickle

class SalesPredictor:
    """
    SalesPredictor class is responsible for loading a pre-trained model and using it to make predictions.
    """

    def __init__(self, model_path):
        """
        Initializes a SalesPredictor instance.

        Parameters:
        model_path (str): The path to the pre-trained model.
        """
        self.model = pickle.load(open(model_path, 'rb'))
        self.input_df = None
        self.original_df = None
        self.prediction = None

    def load_data(self, uploaded_file):
        """
        Loads data from the uploaded CSV file.

        Parameters:
        uploaded_file (str): The uploaded file in CSV format.
        """
        self.input_df = pd.read_csv(uploaded_file, index_col=0)
        self.original_df = self.input_df.copy()
        if 'Total Sales' in self.input_df.columns:
            self.input_df = self.input_df.drop(columns=['Total Sales'])

    def predict(self):
        """
        Uses the loaded model to make predictions based on the input data.

        Returns:
        self.prediction (numpy array): The predicted sales.
        """
        if self.input_df is not None:
            self.prediction = self.model.predict(self.input_df)
        return self.prediction


class SalesApp:
    """
    SalesApp class is responsible for the user interface and calling appropriate SalesPredictor methods.
    """

    def __init__(self):
        """
        Initializes a SalesApp instance.
        """
        self.sales_predictor = SalesPredictor('sales_model.pkl')
        self.uploaded_file = None
        self.toggle_bars_button = False

    def input_features(self):
        """
        Handles user input for the sidebar header and file uploader.
        """
        st.sidebar.header('User Input')
        self.uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    def display_input_data(self):
        """
        Displays the input data uploaded by the user.
        """
        st.subheader('User Input Data')
        if self.uploaded_file is not None:
            self.sales_predictor.load_data(self.uploaded_file)
            st.write(self.sales_predictor.input_df)
        else:
            st.write('No input data yet.')

    def display_prediction(self):
        """
        Displays the sales predictions made by the model.
        """
        prediction = self.sales_predictor.predict()
        if prediction is not None:
            st.subheader('Predicted Sales')
            st.write(prediction)

    def plot_trends(self):
        """
        Plots line charts for sales trends.
        """
        if st.button('Show Trends'):
            st.line_chart(self.sales_predictor.input_df.iloc[1:])

    def plot_bar_charts(self):
        """
        Toggles the visibility of bar charts for sales data.
        """
        self.toggle_bars_button = st.button('Toggle Bar Charts')
        if self.toggle_bars_button:
            df_for_bars = self.sales_predictor.input_df.iloc[1:]
            for column in df_for_bars.columns:
                if pd.api.types.is_numeric_dtype(df_for_bars[column]):
                    st.subheader(f'Bar Chart for {column}')
                    st.bar_chart(df_for_bars[column])

    def compare_predicted_with_actual(self):
        """
        Compares the predicted sales with the actual sales data, if it's available.
        """
        if st.button('Compare Predicted with Actual Sales'):
            if 'Total Sales' in self.sales_predictor.original_df.columns:
                comparison_df = pd.DataFrame()
                comparison_df['Actual Sales'] = self.sales_predictor.original_df['Total Sales']
                comparison_df['Predicted Sales'] = self.sales_predictor.prediction
                comparison_df['Gap (%)'] = ((comparison_df['Actual Sales'] - comparison_df['Predicted Sales']) /
                                            comparison_df['Actual Sales']) * 100
                comparison_df['Gap (%)'] = comparison_df['Gap (%)'].astype(float)
                comparison_df['Gap (%)'] = comparison_df['Gap (%)'].apply(lambda x: "{:.2f}".format(x))
                comparison_df = comparison_df.style.applymap(lambda x: 'color: green' if float(x) > 0 else 'color: red',
                                                             subset=['Gap (%)'])
                st.dataframe(comparison_df)
            else:
                st.write('No "Total Sales" column in the data for comparison.')

    def run(self):
        """
        Runs the SalesApp.
        """
        self.input_features()
        self.display_input_data()
        self.display_prediction()
        self.plot_trends()
        self.plot_bar_charts()
        self.compare_predicted_with_actual()


if __name__ == "__main__":
    app = SalesApp()
    app.run()
