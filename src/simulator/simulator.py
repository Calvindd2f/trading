import pandas as pd
import logging
from datetime import datetime
from src.model import load_model, preprocess_data
from src.utils import execute_trade, calculate_performance_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)

class TradeSimulator:
    """
    Simulates trading based on a given model and dataset

    Attributes:
        model (GradientBoostingClassifier): The model to use for predictions
        data (pd.DataFrame): The dataset to use for trading
        initial_balance (float): The initial balance of the simulator
        trade_amount (float): The amount of money to trade with each time
        balance (float): The current balance of the simulator
        positions (float): The current number of positions held by the simulator
        trade_log (List[dict]): A log of all trades made by the simulator
    """

    def __init__(
        self,
        model_path: str,
        data_path: str,
        initial_balance: float = 10000,
        trade_amount: float = 1000,
    ):
        self.model = load_model(model_path)
        self.data = pd.read_csv(data_path)
        self.balance = initial_balance
        self.trade_amount = trade_amount
        self.positions = 0
        self.trade_log = []
        self.preprocessed_data = preprocess_data(self.data)

    def simulate_trades(self) -> None:
        """
        Simulates trading based on the given model and dataset

        Returns:
            None
        """
        logging.info("Starting trade simulation...")
        for index, row in self.preprocessed_data.iterrows():
            prediction = self.model.predict([row.drop(['time', 'price', 'volume', 'label'])])[0]
            self.execute_trade(predicted=prediction, row=row)
            self.log_trade(index=index, row=row, predicted=prediction)
        self.save_trade_log()

    def execute_trade(
        self,
        predicted: int,
        row: pd.Series,
    ) -> None:
        """
        Executes a trade based on the given prediction and row

        Args:
            predicted (int): The prediction of the model (0: sell, 1: buy)
            row (pd.Series): The row of the dataset to use for the trade

        Returns:
            None
        """
        if predicted == 1:  # Buy signal
            if self.balance >= self.trade_amount:
                self.positions += self.trade_amount / row['price']
                self.balance -= self.trade_amount
                logging.info(
                    f"Buy: {self.trade_amount} at {row['price']}"
                )

        elif predicted == 0 and self.positions > 0:  # Sell signal
            self.balance += self.positions * row['price']
            logging.info(
                f"Sell: {self.positions} at {row['price']}"
            )
            self.positions = 0

    def log_trade(
        self,
        index: int,
        row: pd.Series,
        predicted: int,
    ) -> None:
        """
        Logs a trade

        Args:
            index (int): The index of the row in the dataset
            row (pd.Series): The row of the dataset to use for the trade
            predicted (int): The prediction of the model (0: sell, 1: buy)

        Returns:
            None
        """
        self.trade_log.append(
            {
                'time': row['time'],
                'price': row['price'],
                'predicted': predicted,
                'balance': self.balance,
                'positions': self.positions,
            }
        )

    def save_trade_log(self) -> None:
        """
        Saves the trade log to a CSV file

        Returns:
            None
        """
        trade_log_df = pd.DataFrame(self.trade_log)
        trade_log_df.to_csv('simulation_trade_log.csv', index=False)
        logging.info("Trade log saved to 'simulation_trade_log.csv'")

    def calculate_performance(self) -> dict:
        """
        Calculates the performance metrics of the simulator

        Returns:
            dict: A dictionary containing the performance metrics
        """
        return calculate_performance_metrics(pd.DataFrame(self.trade_log))

def main():
    model_path = 'src/optimized_pump_dump_model.pkl'
    data_path = 'data/historical_data.csv'
    simulator = TradeSimulator(model_path, data_path)
    simulator.simulate_trades()
    performance_metrics = simulator.calculate_performance()
    logging.info(f"Performance Metrics: {performance_metrics}")

if __name__ == "__main__":
    main()
