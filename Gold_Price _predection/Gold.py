from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import tkinter as tk
import numpy as np
from tkinter import filedialog
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

main = tkinter.Tk()
main.title("Gold Price Prediction")
main.geometry("1200x1200")

global filename
global df
global gb_model,rf_model,linear_model
global X_train, X_test, y_train, y_test, X, y
global linear_y_pred,rf_y_pred,gb_y_pred


def upload():
    global filename
    global df
    global X_train, X_test, y_train, y_test,X, y
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    text.delete('1.0', END)
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Dataset Size : "+str(len(df))+"\n")
    text.insert(END, "Sampled dataset size: {}\n".format(df.shape))
    text.insert(END, "Data types:\n{}\n".format(df.dtypes))
    text.insert(END, "Dataset info:\n{}\n".format(df.info()))
    df.dropna(inplace=True)
    # Plot data
    '''fig = px.line(y=df.Price, x=df.Date)
    fig.update_traces(line_color='black') 
    fig.update_layout(xaxis_title="Date", 
                      yaxis_title="Scaled Price",
                      title={'text': "Gold Price History Data", 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor':'top'},
                      plot_bgcolor='rgba(255,223,220,0.8)') 
    fig.show()'''

def convert():
    global X_train, X_test, y_train, y_test, X, y
    # Step 1: Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    # Step 2: Convert price-related columns to numeric format
    price_columns = ['Price', 'Open', 'High', 'Low']
    df[price_columns] = df[price_columns].replace('[\$,]', '', regex=True).astype(float)
    text.insert(END, str(df[price_columns]))
    df['Vol.'] = df['Vol.'].fillna('0')
    text.insert(END, str(df['Vol.']))
    df['Vol.'] = df['Vol.'].replace({'K': '*1e3', 'M': '*1e6'}, regex=True).map(pd.eval).astype(int)
    df['Change %'] = df['Change %'].str.rstrip('%').astype(float)
    text.insert(END, str(df['Change %']))
    X = df.drop(['Date', 'Price'], axis=1)  
    text.insert(END, str(X))
    y = df['Price']  # Target variable is 'Price'
    text.insert(END, str(y))
    
def LR():
    global X_train, X_test, y_train, y_test,X,y
    global linear_y_pred
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_y_pred = linear_model.predict(X_test)
    linear_mse = mean_squared_error(y_test, linear_y_pred)
    text.insert(END, "Mean Squared Error: {}\n".format(linear_mse))

def RF():
    global X_train, X_test, y_train, y_tests,rf_y_pred
    # Initialize and train the random forest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    # Make predictions on the testing set
    rf_y_pred = rf_model.predict(X_test)
    # Evaluate the RandomForestRegressor model
    rf_mse = mean_squared_error(y_test, rf_y_pred)
    text.insert(END, "Random Forest Mean Squared Error: {}\n".format(rf_mse))

def GB():
    global X_train, X_test, y_train, y_test,gb_y_pred
    # Initialize and train the gradient boosting model
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train, y_train)
    # Make predictions on the testing set
    gb_y_pred = gb_model.predict(X_test)
    # Evaluate the GradientBoostingRegressor model
    gb_mse = mean_squared_error(y_test, gb_y_pred)
    text.insert(END, "Gradient Boosting Mean Squared Error: {}\n".format(gb_mse))

def perform():
    global X_train, X_test, y_train, y_test
    global linear_y_pred,rf_y_pred,gb_y_pred
# Calculate evaluation metrics for Linear Regression
    linear_mse = mean_squared_error(y_test, linear_y_pred)
    linear_mae = mean_absolute_error(y_test, linear_y_pred)
    linear_r2 = r2_score(y_test, linear_y_pred)
    # Calculate evaluation metrics for RandomForestRegressor
    rf_mse = mean_squared_error(y_test, rf_y_pred)
    rf_mae = mean_absolute_error(y_test, rf_y_pred)
    rf_r2 = r2_score(y_test, rf_y_pred)
    # Calculate evaluation metrics for GradientBoostingRegressor
    gb_mse = mean_squared_error(y_test, rf_y_pred)
    gb_mae = mean_absolute_error(y_test, gb_y_pred)
    gb_r2 = r2_score(y_test, gb_y_pred)
    # Create a DataFrame to compare the evaluation metrics
    comparison_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
        'Mean Squared Error': [linear_mse, rf_mse, gb_mse],
        'Mean Absolute Error': [linear_mae, rf_mae, gb_mae],
        'R-squared': [linear_r2, rf_r2, gb_r2]
    })
    # Print the comparison DataFrame
    text.insert(END, "\nEvaluation Metrics Comparison:\n")
    text.insert(END, comparison_df.to_string(index=False) + "\n")
    # Create subplots for each evaluation metric
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    # Bar plot for Mean Squared Error (MSE)
    axes[0].bar(comparison_df['Model'], comparison_df['Mean Squared Error'], color=['blue', 'green', 'orange'])
    axes[0].set_title('Mean Squared Error')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('MSE')
    # Bar plot for Mean Absolute Error (MAE)
    axes[1].bar(comparison_df['Model'], comparison_df['Mean Absolute Error'], color=['blue', 'green', 'orange'])
    axes[1].set_title('Mean Absolute Error')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('MAE')
    # Bar plot for R-squared (coefficient of determination)
    axes[2].bar(comparison_df['Model'], comparison_df['R-squared'], color=['blue', 'green', 'orange'])
    axes[2].set_title('R-squared')
    axes[2].set_xlabel('Model')
    axes[2].set_ylabel('R-squared')
    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.show()
# Define the predict function with the necessary arguments
def predict():
    open_price = open_price_entry.get()
    high_price = high_price_entry.get()
    low_price = low_price_entry.get()
    volume = volume_entry.get()
    
        
        




def run_code():
    os.system("python Gold.py")


font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=100,y=200)
text.config(font=font1)

font = ('times', 16, 'bold')
title = Label(main, text='Gold Price Prediction')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=2, width=50)       
title.place(x=400, y=20)

font1 = ('times', 14, 'bold')
upload_button = Button(main, text="Upload Dataset", command=upload)
upload_button.place(x=20, y=80)
upload_button.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=250, y=80)

font1 = ('times', 14, 'bold')
run_button = Button(main, text="Run Gold.py", command=run_code)
run_button.place(x=20, y=140)
run_button.config(font=font1)

font1 = ('times', 14, 'bold')
convert_button = Button(main, text="Convert", command=convert)
convert_button.place(x=250, y=140)
convert_button.config(font=font1)

font1 = ('times', 14, 'bold')
LR_button = Button(main, text="LinearReg", command=LR)
LR_button.place(x=370, y=140)
LR_button.config(font=font1)


font1 = ('times', 14, 'bold')
RF_button = Button(main, text="Random Forest", command=RF)
RF_button.place(x=500, y=140)
RF_button.config(font=font1)


font1 = ('times', 14, 'bold')
GB_button = Button(main, text="Gradient Boosting", command=GB)
GB_button.place(x=700, y=140)
GB_button.config(font=font1)


font1 = ('times', 14, 'bold')
GB_button = Button(main, text="Performance", command=perform)
GB_button.place(x=900, y=140)
GB_button.config(font=font1)

import tkinter as tk
from tkinter import messagebox
import numpy as np

def predict1():
    # Create a new Tkinter window for prediction
    window = tk.Tk()
    window.title("Gold Price Prediction")

    # Function to predict gold price using the best model
    def predict_price():
        try:
            # Get user input from entry widgets
            open_price = float(entry_open_price.get())
            high_price = float(entry_high_price.get())
            low_price = float(entry_low_price.get())
            volume = float(entry_volume.get())
            
            # Perform prediction using the best model
            # Replace this with prediction code using your best model
            # For now, let's use random values for demonstration
            predicted_price = np.random.uniform(1800, 1900)
            
            # Show the predicted price in a message box
            messagebox.showinfo("Prediction Result", f"The predicted gold price is: ${predicted_price:.2f}")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values")

    # Create entry widgets for user input
    label_open_price = tk.Label(window, text="Open Price: (USD)")
    label_open_price.grid(row=0, column=0)
    entry_open_price = tk.Entry(window)
    entry_open_price.grid(row=0, column=1)

    label_high_price = tk.Label(window, text="High Price: (USD)")
    label_high_price.grid(row=1, column=0)
    entry_high_price = tk.Entry(window)
    entry_high_price.grid(row=1, column=1)

    label_low_price = tk.Label(window, text="Low Price: (USD)")
    label_low_price.grid(row=2, column=0)
    entry_low_price = tk.Entry(window)
    entry_low_price.grid(row=2, column=1)

    label_volume = tk.Label(window, text="Volume: (K)")
    label_volume.grid(row=3, column=0)
    entry_volume = tk.Entry(window)
    entry_volume.grid(row=3, column=1)

    # Create a button to trigger prediction
    predict_button = tk.Button(window, text="Predict", command=predict_price)
    predict_button.grid(row=4, columnspan=2)

    # Run the Tkinter event loop
    window.mainloop()

# Add a predict button to your main GUI to call the predict function
font1 = ('times', 14, 'bold')
predict_button = Button(main, text="Predict", command=predict1)
predict_button.place(x=1020, y=140)
predict_button.config(font=font1)



main.config(bg='green')
main.mainloop()

 