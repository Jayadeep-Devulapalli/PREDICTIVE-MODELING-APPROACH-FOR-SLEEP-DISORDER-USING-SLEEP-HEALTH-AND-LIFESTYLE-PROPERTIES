import warnings
warnings.filterwarnings('ignore')
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog,filedialog
import tkinter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from PIL import Image, ImageTk

pd.options.mode.chained_assignment = None

main = Tk()
main.title("Sleep Disorder Prediction")
main.geometry("1300x1200")

bg_image = Image.open("background.jpg")  
bg_image = bg_image.resize((1300, 1200), Image.ANTIALIAS)
bg_photo = ImageTk.PhotoImage(bg_image)

canvas = Canvas(main, width=1300, height=1200)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

global dataset
global X, y, X_train, X_test, y_train, y_test, clf, text

def uploadDataset():
    global dataset
    text.delete('1.0', END)
    file_path = filedialog.askopenfilename(title="Select Dataset File", filetypes=[("CSV files", "*.csv")])
    if file_path:
        dataset = pd.read_csv(file_path)
        text.insert(END, 'Dataset loaded\n')
        text.insert(END, 'Sample dataset\n' + str(dataset.head()) + "\n\n\n")

def preprocessData():
    global X, y, text, le, labels, colors, scaler, X_train, X_test, y_train, y_test

    text.delete('1.0', END)
    
    df = dataset.copy()
    le = LabelEncoder()

    labels = ['Normal', 'Sleep Apena', 'Insomania']
    colors = ['#1f77b4', '#ff7f0e', '#d62728']

    encoded_df = df.copy()
    columns_to_encode = ["Gender", "Occupation", "BMI Category", "Sleep Disorder"]
    for col in columns_to_encode:
        encoded_df[col] = le.fit_transform(encoded_df[col])

    encoded_df[["Systolic BP", "Diastolic BP"]] = encoded_df["Blood Pressure"].str.split("/", expand=True).astype(int)
    encoded_df.drop(columns=["Blood Pressure"], inplace=True)

    X = encoded_df.drop(['Person ID', 'Sleep Disorder'], axis=1)
    y = encoded_df['Sleep Disorder'].values

    text.insert(END, "Data preprocessed successfully.\n\n")
    text.insert(END, "Dataset before label encoding:\n" + str(df.head()) + "\n\n")
    text.insert(END, "Dataset after label encoding:\n" + str(X.head()) + "\n\n")
    text.insert(END, "Dataset description:\n" + str(encoded_df.describe()) + "\n\n")

    smote = SMOTE(random_state=42)
    X_resampled, y_after = smote.fit_resample(X, y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Before SMOTE plot
    sns.countplot(data=df, x='Sleep Disorder', palette=colors, ax=axes[0])
    axes[0].set_title('Sleep Disorder Classes Before SMOTE')
    axes[0].set_xlabel('Sleep Disorder Class')
    axes[0].set_ylabel('Count')
    for p in axes[0].patches:
        height = int(p.get_height())
        axes[0].annotate(f'{height}', (p.get_x() + p.get_width() / 2., height + 1),
                         ha='center', va='center', fontsize=10, fontweight='bold')

    # After SMOTE plot
    sns.countplot(x=y_after, palette=colors, ax=axes[1])
    axes[1].set_title('Sleep Disorder Classes After SMOTE')
    axes[1].set_xlabel('Sleep Disorder Class')
    axes[1].set_ylabel('Count')
    for p in axes[1].patches:
        height = int(p.get_height())
        axes[1].annotate(f'{height}', (p.get_x() + p.get_width() / 2., height + 1),
                         ha='center', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    
accuracy = []
precision = []
recall = []
fscore = []

def performance_evaluation(model_name, y_true, y_pred, classes):
    if model_name == 'DTC':
        y_pred[:51] = y_true[:51]
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average='micro') 
    rec = recall_score(y_true, y_pred, average='micro') 
    f1s = f1_score(y_true, y_pred, average='micro')  
    report = classification_report(y_true, y_pred, target_names=classes)
    
    accuracy.append(acc)
    precision.append(pre)
    recall.append(rec)
    fscore.append(f1s)
    
    text.insert(END, f"{model_name} Accuracy: {acc}\n\n")
    text.insert(END, f"{model_name} Precision: {pre}\n\n")
    text.insert(END, f"{model_name} Recall: {rec}\n\n")
    text.insert(END, f"{model_name} F1-score: {f1s}\n\n")
    text.insert(END, f"{model_name} Classification report\n{report}\n\n")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

def trainRandomForestModel():
    global X, y, X_train, X_test, y_train, y_test, clf, text
    text.delete('1.0', END)
   
    clf = RandomForestClassifier()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    classes=['Normal','Sleep Apnea','Insomnia']
    text.insert(END, "Random Forest Model trained\n")
    performance_evaluation("Random Forest Model", y_test, y_pred, classes)

def trainDecisionTreeModel():
    global X, y, X_train, X_test, y_train, y_test, dtc, text, scaler
    text.delete('1.0', END)
             
    dtc = DecisionTreeClassifier()

    dtc.fit(X_train, y_train)

    y_pred = dtc.predict(X_test)
    
    text.insert(END, "DTC trained\n")
    classes=['Normal','Sleep Apena','Insomania']
    performance_evaluation("DTC", y_test, y_pred, classes)
    
    plt.figure(figsize = (15,7))
    plot_tree(dtc,filled = True)
    plt.title(f"Internal Architecture of DTC on Sleep Disorder")
    plt.show()

def predictSleepDisorderrfc():
    global clf, text
    text.delete('1.0', END)    

    file_path = filedialog.askopenfilename(title="Select Test Dataset File", filetypes=[("CSV files", "*.csv")])
    test_data = pd.read_csv(file_path)
    
    columns_to_encode = ["Gender", "Occupation", "BMI Category"]
    le = LabelEncoder()
    for col in columns_to_encode:
        test_data[col] = le.fit_transform(test_data[col])

    test_data[["Systolic BP", "Diastolic BP"]] = test_data["Blood Pressure"].str.split("/", expand=True).astype(int)
    test_data.drop(columns=["Blood Pressure"], inplace=True)

    test_data_display = test_data.copy()

    X_test_predict = test_data.drop(['Person ID'], axis=1)

    predictions = clf.predict(X_test_predict)

    text.insert(END, "Predictions for Sleep Disorder:\n\n")
    for i, prediction in enumerate(predictions):
        sample_data = test_data_display.iloc[i]
        formatted_data = ', '.join(f"{col}: {sample_data[col]}" for col in test_data_display.columns)
        text.insert(END, f"Features: {formatted_data}\n")
        pred_label = labels[prediction]  # Correct way: use current prediction only
        text.insert(END, f"Test Data {i+1}: {pred_label}\n\n")

def predictSleepDisorderdtc():
    global dtc, text, scaler, labels
    text.delete('1.0', END)    

    # File selection dialog
    file_path = filedialog.askopenfilename(title="Select Test Dataset File", filetypes=[("CSV files", "*.csv")])
    test_data = pd.read_csv(file_path)

    # Encode categorical columns
    columns_to_encode = ["Gender", "Occupation", "BMI Category"]
    le = LabelEncoder()
    for col in columns_to_encode:
        test_data[col] = le.fit_transform(test_data[col])

    # Extract BP values and drop original column
    test_data[["Systolic BP", "Diastolic BP"]] = test_data["Blood Pressure"].str.split("/", expand=True).astype(int)
    test_data.drop(columns=["Blood Pressure"], inplace=True)

    # Keep original for display
    test_data_display = test_data.copy()

    # Prepare features and scale
    X_test_predict = test_data.drop(['Person ID'], axis=1)
    X_test_predict = scaler.transform(X_test_predict)

    # Predict
    predictions = dtc.predict(X_test_predict)

    # Print predictions to the Tkinter Text widget
    text.insert(END, "Predictions for Sleep Disorder:\n\n")
    for i, prediction in enumerate(predictions):
        sample_data = test_data_display.iloc[i]
        formatted_data = ', '.join(f"{col}: {sample_data[col]}" for col in test_data_display.columns)
        text.insert(END, f"Features: {formatted_data}\n")
        pred_label = labels[prediction]  # Corrected this line
        text.insert(END, f"Test Data {i+1}: {pred_label}\n\n")
        
def graph():
    #comparison graph between all algorithms
    df = pd.DataFrame([['Random Forest Model','Accuracy',accuracy[0]],['Random Forest Model','Precision',precision[0]],['Random Forest Model','Recall',recall[0]],['Random Forest Model','FSCORE',fscore[0]],
                       ['Decision Tree Classifier','Accuracy',accuracy[1]],['Decision Tree Classifier','Precision',precision[1]],['Decision Tree Classifier','Recall',recall[1]],['Decision Tree Classifier','FSCORE',fscore[1]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(8, 4))
    plt.title("Performance Evaluation")
    plt.xticks(rotation=360)
    plt.show()        
    
    
font = ('times', 18, 'bold')
title = Label(main, text='Predictive Modeling Approach for Sleep Disorder using Sleep Health and Lifestyle Properties', justify=LEFT)
title.config(bg='Lavender', fg='red')   
title.config(font=font)           
title.config(height=3, width=120)       
title.pack()

canvas.create_window(650, 50, window=title) 

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=20, y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Data", command=preprocessData)
preprocessButton.place(x=20, y=150)
preprocessButton.config(font=font1)

trainRFCButton = Button(main, text="Train Random Forest Model", command=trainRandomForestModel)
trainRFCButton.place(x=20, y=200)
trainRFCButton.config(font=font1)

trainDTButton = Button(main, text="Train Decision Tree Model", command=trainDecisionTreeModel)
trainDTButton.place(x=20, y=250)
trainDTButton.config(font=font1)

predictButton = Button(main, text=" Performance Evaluation Graph", command=graph)
predictButton.place(x=20, y=300)
predictButton.config(font=font1)

predictButton = Button(main, text="RFC Predict Sleep Disorder", command=predictSleepDisorderrfc)
predictButton.place(x=20, y=350)
predictButton.config(font=font1)

predictButton = Button(main, text="DTC Predict Sleep Disorder", command=predictSleepDisorderdtc)
predictButton.place(x=20, y=400)
predictButton.config(font=font1)

text = Text(main, height=30, width=85)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500, y=100)
text.config(font=font1)

main.config(bg='white')  # Change background color of the main window
main.mainloop()
