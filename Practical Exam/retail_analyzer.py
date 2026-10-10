
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


FILE = "data/retail_sales.csv"
CHART_DIR = "charts"


# Abstraction
class Person(ABC):

    def __init__(self, product, category):
        self.product = product
        self.category = category

    @abstractmethod
    def display_summary(self):
        pass

# Inheritance, Encapsulation and polymorphism
class Sales(Person):

    def __init__(self, product, category, total_sales, price):
        super().__init__(product, category)
        self.total_sales = total_sales
        self.__price = price  # Private Attribute

    # Getter
    def get_price(self):
        return self.__price

    # Setter
    def set_price(self, price):
        if price > 0:
            self.__price = price
        else:
            print("Price must be positive!")

    # Polymophism
    def display_summary(self):
        print("\n========== SALES DETAILS ==========")
        print("Product:", self.product)
        print("Category:", self.category)
        print("Total Sales:", self.total_sales)
        print("Price:", self.get_price())


class SalesAnalysis:

    def __init__(self):
        self.df = pd.DataFrame()

    # Load CSV
    def load_csv(self):

        if not os.path.exists(FILE):
            print("CSV file not found:", FILE)
            return False

        try:
            self.df = pd.read_csv(FILE)

            required = ["Date", "Product", "Category","Price", "Quantity_Sold", "Total_Sales"]

            missing = [col for col in required if col not in self.df.columns]

            if missing:
                print("Missing columns:", missing)
                return False

            print("CSV loaded successfully!")
            print("Dataset shape:", self.df.shape)
            print(self.df.head())

            return True

        except (OSError, pd.errors.ParserError) as e:
            print("Error loading CSV:", e)
            return False

    # Clean data
    def clean_data(self):

        self.df.drop_duplicates(inplace=True)

        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")

        for col in ["Price", "Quantity_Sold", "Total_Sales"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self.df.dropna(subset=["Date", "Product", "Category","Price", "Quantity_Sold", "Total_Sales"],inplace=True)

        self.df = self.df[(self.df["Price"] > 0) & (self.df["Quantity_Sold"] > 0) & (self.df["Total_Sales"] >= 0)].copy()

        # Revenue column for Analysis 
        self.df["Revenue"] = self.df["Total_Sales"]

        print("\nData cleaned successfully!")
        print("Total records:", len(self.df))

    # NumPy and Pandas calculations
    def calculate_columns(self):

        revenue = self.df["Revenue"].to_numpy()

        print("\n========== SALES STATISTICS ==========")
        print("Total Revenue:", round(np.sum(revenue), 2))
        print("Average Revenue:", round(np.mean(revenue), 2))
        print("Maximum Revenue:", round(np.max(revenue), 2))
        print("Minimum Revenue:", round(np.min(revenue), 2))

        print("\nRevenue by Product:")
        print(self.df.groupby("Product")["Revenue"].sum())

        print("\nRevenue by Category:")
        print(self.df.groupby("Category")["Revenue"].sum())

        print("\nSales Counts:")
        print(self.df["Total_Sales"].value_counts())

        print("\nFirst Five Records:")
        print(self.df.head())

    # filter 
    def filter_category(self):

        print("\n========== FILTER SALES ==========")
        print("1. Filter by Product")
        print("2. Filter by Category")
        print("3. Show Top 10 Orders")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            product = input("Enter product name: ").strip()
            result = self.df[self.df["Product"].str.lower() == product.lower()]

        elif choice == "2":
            category = input("Enter category name: ").strip()
            result = self.df[self.df["Category"].str.lower() == category.lower()]

        elif choice == "3":
            result = self.df.nlargest(10, "Revenue")

        else:
            print("Invalid choice!")
            return

        if result.empty:
            print("No matching records found.")
        else:
            print(result.to_string(index=False))

    # Generate Report 
    def generate_report(self):

        print("\n========== SALES REPORT ==========")
        print("Total Orders:", len(self.df))
        print("Total Revenue:", round(self.df["Revenue"].sum(), 2))

        print("\nRevenue by Category:")
        print(self.df.groupby("Category")["Revenue"].sum())

        print("\nTop 10 Orders:")
        print(self.df.nlargest(10, "Revenue"))

        self.df.to_csv("sales_report.csv", index=False)
        print("\nReport saved as sales_report.csv")

    # Data visualization
    def show_charts(self):

        os.makedirs(CHART_DIR, exist_ok=True)

        print("\n========== CHART MENU ==========")
        print("1. Bar Chart - Revenue by Product")
        print("2. Pie Chart - Revenue by Category")
        print("3. Line Chart - Monthly Revenue")
        print("4. Heatmap - Numeric Correlation")
        print("5. Show All Charts")

        choice = input("Enter choice: ").strip()

        if choice not in ["1", "2", "3", "4", "5"]:
            print("Invalid chart choice!")
            return

        # Bar chart
        if choice in ["1", "5"]:

            data = self.df.groupby("Category")["Revenue"].sum()

            plt.figure(figsize=(14, 5))
            plt.bar(data.index, data.values)
            plt.title("Revenue by Category")
            plt.xlabel("Category")
            plt.ylabel("Revenue")
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(os.path.join(CHART_DIR, "bar.png"))
            
            
            plt.show()
            plt.close()

        # Pie chart
        if choice in ["2", "5"]:

            data = self.df.groupby("Category")["Revenue"].sum()

            if data.sum() > 0:
                plt.figure(figsize=(8, 5))
                plt.pie(
                    data.values,
                    labels=data.index,
                    autopct="%1.1f%%"
                )
                plt.title("Revenue by Category")
                plt.tight_layout()
                plt.savefig(os.path.join(CHART_DIR, "pie.png"))
                plt.legend()
                plt.show()
                plt.close()
            else:
                print("No positive revenue to display.")

        # Line chart
        if choice in ["3", "5"]:

            data = self.df.copy()
            data["Month"] = data["Date"].dt.to_period("M")

            monthly = data.groupby("Month")["Revenue"].sum()

            plt.figure(figsize=(9, 5))
            plt.plot(monthly.index.astype(str),
                     monthly.values, marker="o")
            
            plt.title("Monthly Revenue")
            plt.xlabel("Month")
            plt.ylabel("Revenue")
            plt.xticks(rotation=30)
            plt.tight_layout()
            
            plt.savefig(os.path.join(CHART_DIR, "line.png"))
            plt.show()
            plt.close()

        # Heatmap
        if choice in ["4", "5"]:

            numeric = self.df[
                ["Total_Sales", "Price", "Quantity_Sold"]
            ]

            plt.figure(figsize=(8, 5))
            sns.heatmap(
                numeric.corr(),
                annot=True,
                cmap="coolwarm"
            )
            plt.title("Sales Correlation Heatmap")
            plt.tight_layout()
            plt.legend()
            plt.savefig(os.path.join(CHART_DIR, "heatmap.png"))
            plt.show()
            plt.close()

        if choice == "5":
            print("All charts saved in the charts folder.")

def main():

    analysis = SalesAnalysis()

    if not analysis.load_csv():
        return

    analysis.clean_data()

    if analysis.df.empty:
        print("No valid sales records.")
        return

    analysis.calculate_columns()


    row = analysis.df.iloc[0]

    sale = Sales(row["Product"],row["Category"],row["Total_Sales"],row["Price"])

    sale.display_summary()

    # Main menu
    while True:

        print("\n========== SALES ANALYTICS DASHBOARD ==========")
        print("1. Display Sales Data")
        print("2. Calculate Statistics")
        print("3. Filter Sales")
        print("4. Generate Report")
        print("5. Show Charts")
        print("6. Exit")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            print(analysis.df.to_string(index=False))

        elif choice == "2":
            analysis.calculate_columns()

        elif choice == "3":
            analysis.filter_category()

        elif choice == "4":
            analysis.generate_report()

        elif choice == "5":
            analysis.show_charts()

        elif choice == "6":
            print("Thank you for using Sales Analytics Dashboard!")
            break

        else:
            print("Invalid choice! Try again.")


if __name__ == "__main__":
    main()
