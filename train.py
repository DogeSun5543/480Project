import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

class SalesModel:
    """
    SalesModel is a class that represents a random forest regressor model for sales prediction.
    """

    def __init__(self, train_data_file, test_data_file):
        """
        Initializes the SalesModel object.

        Args:
            train_data_file (str): Path to the CSV file containing the training data.
            test_data_file (str): Path to the CSV file containing the test data.
        """
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.model = None

    def load_data(self):
        """
        Loads the training and test data from CSV files.
        """
        self.train_data = pd.read_csv(self.train_data_file, index_col=0)
        self.test_data = pd.read_csv(self.test_data_file, index_col=0)

    def prepare_data(self):
        """
        Prepares the training and test data by separating the target variable.
        """
        self.train_y = self.train_data['Total Sales']
        self.train_X = self.train_data.drop(['Total Sales'], axis=1)

        self.test_y = self.test_data['Total Sales']
        self.test_X = self.test_data.drop(['Total Sales'], axis=1)

    def train_model(self):
        """
        Trains the random forest regressor model using the training data.
        """
        self.model = RandomForestRegressor(random_state=1)
        self.model.fit(self.train_X, self.train_y)

    def save_model(self, file_name):
        """
        Saves the trained model as a pickle file.

        Args:
            file_name (str): Path to save the pickle file.
        """
        pickle.dump(self.model, open(file_name, 'wb'))


# Create an instance of SalesModel
sales_model = SalesModel("MarData.csv", "AprData.csv")

# Load and prepare the data
sales_model.load_data()
sales_model.prepare_data()

# Train the model
sales_model.train_model()

# Save the trained model
sales_model.save_model("sales_model.pkl")
