import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, Entry, messagebox, ttk, Listbox, MULTIPLE, Scrollbar, LEFT, Y, RIGHT
import pyperclip
from tkinter import ttk
import os

class SalesForecastApp:
    def __init__(self, master):
        self.master = master
        master.title("Sales Forecasting with Prophet")

        self.label = Label(master, text="Sales Forecasting Application")
        self.label.pack()

        # Input field for number of past months
        self.past_months_label = Label(master, text="Number of Past Months to Use:")
        self.past_months_label.pack()

        self.past_months_entry = Entry(master)
        self.past_months_entry.pack()
        self.past_months_entry.insert(0, "12")  # Default to the last 12 months

        # Input field for number of future months
        self.future_months_label = Label(master, text="Number of Future Months to Forecast:")
        self.future_months_label.pack()

        self.future_months_entry = Entry(master)
        self.future_months_entry.pack()
        self.future_months_entry.insert(0, "12")  # Default to 12 months

        # Input field for selecting the start date
        self.start_date_label = Label(master, text="Starting Date for Forecast (YYYY-MM-DD):")
        self.start_date_label.pack()

        self.start_date_entry = Entry(master)
        self.start_date_entry.pack()
        self.start_date_entry.insert(0, "2023-01-01")  # Default start date

        # Label for selecting outlier months
        self.outlier_label = Label(master, text="Select Outlier Months (1=Jan, 2=Feb, ...). Outlier months to exclude from calculation.")
        self.outlier_label.pack()

        # Frame to hold the listbox and scrollbar
        outlier_frame = ttk.Frame(master)
        outlier_frame.pack()

        # Listbox for selecting outlier months (represented by numbers)
        self.outlier_listbox = Listbox(outlier_frame, selectmode=MULTIPLE, height=6)
        self.outlier_listbox.pack(side=LEFT, fill=Y)

        # Scrollbar for the outlier listbox
        outlier_scrollbar = Scrollbar(outlier_frame, orient="vertical")
        outlier_scrollbar.pack(side=RIGHT, fill=Y)

        # Link the scrollbar to the outlier listbox
        self.outlier_listbox.config(yscrollcommand=outlier_scrollbar.set)
        outlier_scrollbar.config(command=self.outlier_listbox.yview)

        # Populate the listbox with months
        for i in range(1, 13):
            self.outlier_listbox.insert("end", str(i))

        # Frame for the Treeview and its scrollbar
        tree_frame = ttk.Frame(master)
        tree_frame.pack()

        # Treeview widget for displaying forecast data
        self.tree = ttk.Treeview(tree_frame, columns=("Date", "Predicted Sales", "Lower Bound", "Upper Bound"), show='headings')
        self.tree.heading("Date", text="Date")
        self.tree.heading("Predicted Sales", text="Predicted Sales")
        self.tree.heading("Lower Bound", text="Lower Bound")
        self.tree.heading("Upper Bound", text="Upper Bound")
        self.tree.pack(side=LEFT, fill=Y)

        # Scrollbar for the Treeview
        tree_scrollbar = Scrollbar(tree_frame, orient="vertical")
        tree_scrollbar.pack(side=RIGHT, fill=Y)

        # Link the scrollbar to the Treeview
        self.tree.config(yscrollcommand=tree_scrollbar.set)
        tree_scrollbar.config(command=self.tree.yview)

        self.plot_button = Button(master, text="Show Forecast Plot", command=self.show_forecast_plot)
        self.plot_button.pack()

        self.components_button = Button(master, text="Show Components Plot", command=self.show_components_plot)
        self.components_button.pack()

        # Button for pasting values
        self.paste_values_button = Button(master, text="Paste Values", command=self.paste_values)
        self.paste_values_button.pack()

        # Button for exporting to Excel
        self.export_button = Button(master, text="Export to Excel", command=self.export_to_excel)
        self.export_button.pack()

        self.data = {'ds': pd.DatetimeIndex([]), 'y': []}

    def generate_dates(self, start_date, num_months):
        dates = pd.date_range(start=start_date, periods=num_months, freq='MS')
        return dates

    def paste_values(self):
        clipboard = pyperclip.paste()
        values = clipboard.split()

        try:
            # Get the starting date and number of past months from the user input
            start_date = pd.to_datetime(self.start_date_entry.get())
            num_past_months = int(self.past_months_entry.get())

            if num_past_months <= 0:
                raise ValueError("Number of past months must be a positive integer.")

            # Generate dates based on the starting date and number of past months
            self.data['ds'] = self.generate_dates(start_date, num_past_months)
            self.data['y'] = list(map(float, values))

            if len(self.data['y']) != num_past_months:
                raise ValueError(f"Number of sales values pasted ({len(self.data['y'])}) does not match the number of past months ({num_past_months}).")

            print(f"Generated Dates: {self.data['ds']}")
            print(f"Pasted Values: {self.data['y']}")

            self.run_sales_forecast_if_ready()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse values: {e}")

    def get_selected_outliers(self):
        selected_indices = self.outlier_listbox.curselection()
        selected_months = [int(self.outlier_listbox.get(i)) for i in selected_indices]
        return selected_months

    def run_sales_forecast_if_ready(self):
        if not self.data['ds'].empty and self.data['y']:
            self.run_sales_forecast()

    def run_sales_forecast(self):
        try:
            # Clear previous treeview data
            for i in self.tree.get_children():
                self.tree.delete(i)

            # Get the number of future months from the entry
            try:
                future_months = int(self.future_months_entry.get())
                if future_months <= 0:
                    raise ValueError("Number of future months must be a positive integer.")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid number of future months: {e}")
                return

            # Get the selected outlier months
            outlier_months = self.get_selected_outliers()

            # Exclude outlier months from the data
            df = pd.DataFrame(self.data)
            df['month'] = df['ds'].dt.month
            df = df[~df['month'].isin(outlier_months)]  # Exclude outliers

            # Drop the 'month' column as it's no longer needed
            df = df.drop(columns=['month'])

            # Initialize the Prophet model
            self.model = Prophet()

            # Fit the model on the filtered data
            self.model.fit(df)

            # Create a DataFrame to hold future dates based on user input
            self.future = self.model.make_future_dataframe(periods=future_months, freq='M')

            # Predict future sales
            self.forecast = self.model.predict(self.future)

            # Insert forecast data into the Treeview
            for index, row in self.forecast.tail(future_months).iterrows():  # Displaying the last predicted months
                self.tree.insert("", "end", values=(
                    row['ds'].strftime('%Y-%m-%d'),
                    f"{int(round(row['yhat']))}",
                    f"{int(round(row['yhat_lower']))}",
                    f"{int(round(row['yhat_upper']))}"
                ))

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_forecast_plot(self):
        try:
            # Plot the forecast with annotations
            future_months = int(self.future_months_entry.get())
            fig = self.model.plot(self.forecast)

            # Add annotations for predicted points
            for i, row in self.forecast.tail(future_months).iterrows():
                plt.annotate(f"{int(round(row['yhat']))}", (row['ds'], row['yhat']),
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='red')

            plt.title("Sales Forecast")
            plt.xlabel("Date")
            plt.ylabel("Sales")

            # Adjust layout to prevent title cutoff
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_components_plot(self):
        try:
            # Plot the forecast components (trend, weekly seasonality, yearly seasonality)
            self.model.plot_components(self.forecast)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_to_excel(self):
        try:
            # Combine the original data and the forecast data
            df_past = pd.DataFrame(self.data)
            df_past.columns = ['Date', 'Sales']

            # Get the number of future months from the entry
            future_months = int(self.future_months_entry.get())

            # Only take the last 'future_months' predictions from the forecast
            df_future = self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_months)
            df_future.columns = ['Date', 'Predicted Sales', 'Lower Bound', 'Upper Bound']

            # Round the numbers to no decimal places
            df_past['Sales'] = df_past['Sales'].round(0).astype(int)
            df_future['Predicted Sales'] = df_future['Predicted Sales'].round(0).astype(int)
            df_future['Lower Bound'] = df_future['Lower Bound'].round(0).astype(int)
            df_future['Upper Bound'] = df_future['Upper Bound'].round(0).astype(int)

            # Concatenate past sales with the selected predicted future sales
            df_combined = pd.concat([df_past, df_future], ignore_index=True)

            # Define the file path
            file_path = 'C:/Users/Frank/Desktop/forecasted_sales.xlsx'

            # Export the combined DataFrame to Excel
            df_combined.to_excel(file_path, index=False)

            messagebox.showinfo("Export Success", f"Forecast results successfully exported to {file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export to Excel: {e}")

# Run the Tkinter application
root = Tk()
app = SalesForecastApp(root)
root.mainloop()
