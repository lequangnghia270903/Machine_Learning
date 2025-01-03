import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import pandas as pdw
import plotly.express as px
from matplotlib import pyplot as plt
from cmath import nan
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from helper import data, seconddata, match_elements, describe, outliers, drop_items, download_data, filter_data, num_filter_data, rename_columns, clear_image_cache, handling_missing_values, data_wrangling
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


st.set_page_config(
     page_title="Machine Learning - Data Analysis App",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
)

st.sidebar.title("Machine Learning - Data Analysis Web App")
st.sidebar.info("Đồ Án Môn Học Học Phần Học Máy (5).")  
st.sidebar.info("Nhóm 19_Lê Quang Nghĩa _ Võ Khắc Đoài")

file_format_type = ["csv", "txt", "xls", "xlsx", "ods", "odt"]
functions = ["Overview", "Outliers", "Drop Columns", "Drop Categorical Rows", "Drop Numeric Rows", "Rename Columns", "Display Plot", "Linear Regression","Logistic Regression","Decision Tree","KNN","Handling Missing Data", "Data Wrangling"]
excel_type =["vnd.ms-excel","vnd.openxmlformats-officedocument.spreadsheetml.sheet", "vnd.oasis.opendocument.spreadsheet", "vnd.oasis.opendocument.text"]

uploaded_file = st.sidebar.file_uploader("Upload Your file", type=file_format_type)

if uploaded_file is not None:
    st.title("Đồ Án Kết Thúc Môn Học Học Phần Học Máy (5).")
    st.info("Nhóm 19_Lê Quang Nghĩa _ Võ Khắc Đoài") 
    file_type = uploaded_file.type.split("/")[1]
    
    if file_type == "plain":
        seperator = st.sidebar.text_input("Please Enter what seperates your data: ", max_chars=5) 
        data = data(uploaded_file, file_type,seperator)

    elif file_type in excel_type:
        data = data(uploaded_file, file_type)

    else:
        data = data(uploaded_file, file_type)
    
    describe, shape, columns, num_category, str_category, null_values, dtypes, unique, str_category, column_with_null_values = describe(data)

    multi_function_selector = st.sidebar.multiselect("Enter Name or Select the Column which you Want To Plot: ",functions, default=["Overview"])

    st.subheader("Dataset Preview")
    st.dataframe(data)

    st.text(" ")
    st.text(" ")
    st.text(" ")

    if "Overview" in multi_function_selector:
        st.subheader("Dataset Description")
        st.write(describe)

        st.text(" ")
        st.text(" ")
        st.text(" ")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.text("Basic Information")
            st.write("Dataset Name")
            st.text(uploaded_file.name)

            st.write("Dataset Size(MB)")
            number = round((uploaded_file.size*0.000977)*0.000977,2)
            st.write(number)

            st.write("Dataset Shape")
            st.write(shape)
            
        with col2:
            st.text("Dataset Columns")
            st.write(columns)
        
        with col3:
            st.text("Numeric Columns")
            st.dataframe(num_category)
        
        with col4:
            st.text("String Columns")
            st.dataframe(str_category)
            

        col5, col6, col7, col8= st.columns(4)

        with col6:
            st.text("Columns Data-Type")
            st.dataframe(dtypes)
        
        with col7:
            st.text("Counted Unique Values")
            st.write(unique)
        
        with col5:
            st.write("Counted Null Values")
            st.dataframe(null_values)

# ==================================================================================================
    if "Outliers" in multi_function_selector:

        outliers_selection = st.multiselect("Enter or select Name of the columns to see Outliers:", num_category)
        outliers = outliers(data, outliers_selection)
        
        for i in range(len(outliers)):
            st.image(outliers[i])
# ===================================================================================================

    if "Drop Columns" in multi_function_selector:
        
        multiselected_drop = st.multiselect("Please Type or select one or Multipe Columns you want to drop: ", data.columns)
        
        droped = drop_items(data, multiselected_drop)
        st.write(droped)
        
        drop_export = download_data(droped, label="Droped(edited)")

# =====================================================================================================================================
    if "Drop Categorical Rows" in multi_function_selector:

        filter_column_selection = st.selectbox("Please Select or Enter a column Name: ", options=data.columns)
        filtered_value_selection = st.multiselect("Enter Name or Select the value which you don't want in your {} column(You can choose multiple values): ".format(filter_column_selection), data[filter_column_selection].unique())
        
        filtered_data = filter_data(data, filter_column_selection, filtered_value_selection)
        st.write(filtered_data)
        
        filtered_export = download_data(filtered_data, label="filtered")

# =============================================================================================================================

    if "Drop Numeric Rows" in multi_function_selector:

        option = st.radio(
        "Which kind of Filteration you want",
        ('Delete data inside the range', 'Delete data outside the range'))

        num_filter_column_selection = st.selectbox("Please Select or Enter a column Name: ", options=num_category)
        selection_range = data[num_filter_column_selection].unique()

        for i in range(0, len(selection_range)) :
            selection_range[i] = selection_range[i]
        selection_range.sort()

        selection_range = [x for x in selection_range if np.isnan(x) == False]

        start_value, end_value = st.select_slider(
        'Select a range of Numbers you want to edit or keep',
        options=selection_range,
        value=(min(selection_range), max(selection_range)))
        
        if option == "Delete data inside the range":
            st.write('We will be removing all the values between ', int(start_value), 'and', int(end_value))
            num_filtered_data = num_filter_data(data, start_value, end_value, num_filter_column_selection, param=option)
        else:
            st.write('We will be Keeping all the values between', int(start_value), 'and', int(end_value))
            num_filtered_data = num_filter_data(data, start_value, end_value, num_filter_column_selection, param=option)

        st.write(num_filtered_data)
        num_filtered_export = download_data(num_filtered_data, label="num_filtered")


# =======================================================================================================================================

    if "Rename Columns" in multi_function_selector:

        if 'rename_dict' not in st.session_state:
            st.session_state.rename_dict = {}

        rename_dict = {}
        rename_column_selector = st.selectbox("Please Select or Enter a column Name you want to rename: ", options=data.columns)
        rename_text_data = st.text_input("Enter the New Name for the {} column".format(rename_column_selector), max_chars=50)


        if st.button("Draft Changes", help="when you want to rename multiple columns/single column  so first you have to click Save Draft button this updates the data and then press Rename Columns Button."):
            st.session_state.rename_dict[rename_column_selector] = rename_text_data
        st.code(st.session_state.rename_dict)

        if st.button("Apply Changes", help="Takes your data and rename the column as your wish."):
            rename_column = rename_columns(data, st.session_state.rename_dict)
            st.write(rename_column)
            export_rename_column = download_data(rename_column, label="rename_column")
            st.session_state.rename_dict = {}

# ===================================================================================================================
 
    if "Display Plot" in multi_function_selector:

        multi_bar_plotting = st.multiselect("Enter Name or Select the Column which you Want To Plot: ", str_category)
        
        for i in range(len(multi_bar_plotting)):
            column = multi_bar_plotting[i]
            st.markdown("#### Bar Plot for {} column".format(column))
            bar_plot = data[column].value_counts().reset_index().sort_values(by=column, ascending=False)
            st.bar_chart(bar_plot)

# ==================================================================================================================== 
    # Xác định tất cả các biến phân loại và định lượng
    categorical_variables = [col for col in data.columns if data[col].dtype == 'object']
    quantitative_variables = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]

    # Hiển thị tiêu đề
    st.title("Statistics And Graphing All Possible Cases.")
    case = st.selectbox("Select case:", options=["1 Categorical", "1 Quantitative", "2 Categorical", "1 Categorical-1 Quantitative", "2 Quantitative"])

    # Trường hợp 1: Biến phân loại (1 Categorical)
    if case == "1 Categorical":
        st.header("1 Categorical Variable")
        x_variable = st.selectbox("Select categorical variable:", options=categorical_variables)
        y_variable = None  # Không cần chọn biến y
        if x_variable:
            fig = px.histogram(data, x=x_variable)
            st.plotly_chart(fig)

    # Trường hợp 2: Biến định lượng (1 Quantitative)
    elif case == "1 Quantitative":
        st.header("1 Quantitative Variable")
        y_variable = st.selectbox("Select quantitative variable:", options=quantitative_variables)
        x_variable = None  # Không cần chọn biến x
        if y_variable:
            fig = px.histogram(data, x=y_variable)
            st.plotly_chart(fig)

    # Trường hợp 3: Hai biến phân loại (2 Categorical)
    elif case == "2 Categorical":
        st.header("2 Categorical Variables")
        x_variable = st.selectbox("Select first categorical variable:", options=categorical_variables)
        y_variable = st.selectbox("Select second categorical variable:", options=categorical_variables)
        if x_variable and y_variable:
            fig = px.histogram(data, x=x_variable, color=y_variable)
            st.plotly_chart(fig)

    # Trường hợp 4: Một biến phân loại và một biến định lượng (1 Categorical - 1 Quantitative)
    elif case == "1 Categorical-1 Quantitative":
        st.header("1 Categorical - 1 Quantitative Variables")
        x_variable = st.selectbox("Select categorical variable:", options=categorical_variables)
        y_variable = st.selectbox("Select quantitative variable:", options=quantitative_variables)
        if x_variable and y_variable:
            fig = px.box(data, x=x_variable, y=y_variable)
            st.plotly_chart(fig)

    # Trường hợp 5: Hai biến định lượng (2 Quantitative)
    elif case == "2 Quantitative":
        st.header("2 Quantitative Variables")
        x_variable = st.selectbox("Select first quantitative variable:", options=quantitative_variables)
        y_variable = st.selectbox("Select second quantitative variable:", options=quantitative_variables)
        if x_variable and y_variable:
            fig = px.scatter(data, x=x_variable, y=y_variable)
            st.plotly_chart(fig)
# ====================================================================================================================
    # if "Linear Regression".strip() in multi_function_selector:
    #     # Yêu cầu người dùng chọn biến độc lập (x) và biến phụ thuộc (y)
    #     x_columns = st.multiselect("Select the independent variable(s) (x):", options=data.columns)
    #     y_column = st.selectbox("Select the dependent variable (y):", options=data.columns)

    #     # Kiểm tra nếu người dùng đã chọn các biến
    #     if x_columns and y_column:
    #         # Lấy dữ liệu của biến độc lập (x) và biến phụ thuộc (y)
    #         X = data[x_columns]
    #         y = data[[y_column]]  # Chuyển đổi Series thành DataFrame với một cột

    #         # Xử lý dữ liệu bị thiếu trong biến phụ thuộc (y)
    #         imputer = SimpleImputer(strategy='mean')
    #         y = imputer.fit_transform(y)

    #         # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #         # Xây dựng mô hình hồi quy tuyến tính đa biến
    #         model = LinearRegression()
    #         model.fit(X_train, y_train)

    #         # Dự đoán kết quả và đánh giá mô hình
    #         y_pred = model.predict(X_test)
    #         mse = mean_squared_error(y_test, y_pred)
    #         st.write("Mean Squared Error:", mse)

    #         # Vẽ biểu đồ scatter plot cho từng biến độc lập và biến phụ thuộc trên cùng một biểu đồ
    #         fig, ax = plt.subplots()
    #         for column in x_columns:
    #             ax.scatter(X_test[column], y_test, label=column)
    #         ax.set_xlabel("Independent Variables")
    #         ax.set_ylabel("Dependent Variable")
    #         ax.legend()
    #         st.pyplot(fig)

    if "Linear Regression".strip() in multi_function_selector:
        # Yêu cầu người dùng chọn biến độc lập (x) và biến phụ thuộc (y)
        x_columns = st.multiselect("Select the independent variable(s) (x):", options=data.columns)
        y_column = st.selectbox("Select the dependent variable (y):", options=data.columns)

        # Kiểm tra nếu người dùng đã chọn các biến
        if x_columns and y_column:
            # Lấy dữ liệu của biến độc lập (x) và biến phụ thuộc (y)
            X = data[x_columns]
            y = data[[y_column]]  # Chuyển đổi Series thành DataFrame với một cột

            # Kiểm tra và mã hóa biến phụ thuộc (y) nếu cần thiết
            if y[y_column].dtype == 'object' or isinstance(y.iloc[0, 0], str):
                le = LabelEncoder()
                y = le.fit_transform(y[y_column].astype(str)).reshape(-1, 1)
            else:
                y = y.astype(float)  # Đảm bảo y là kiểu float cho hồi quy tuyến tính

            # Xử lý dữ liệu bị thiếu trong biến phụ thuộc (y)
            imputer = SimpleImputer(strategy='mean')
            y = imputer.fit_transform(y)
            
            # Mã hóa các biến độc lập (x) nếu cần thiết
            X_encoded = pd.get_dummies(X, drop_first=True)

            # Lưu lại tên các cột đã được mã hóa
            encoded_columns = X_encoded.columns

            # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

            # Xây dựng mô hình hồi quy tuyến tính đa biến
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Dự đoán kết quả và đánh giá mô hình
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.write("Mean Squared Error:", mse)

            # Vẽ biểu đồ scatter plot cho các biến đã được mã hóa và biến phụ thuộc trên cùng một biểu đồ
            fig, ax = plt.subplots()
            for column in encoded_columns:
                ax.scatter(X_test[column], y_test, label=column)
            ax.set_xlabel("Independent Variables")
            ax.set_ylabel("Dependent Variable")
            ax.legend()
            st.pyplot(fig)
# ====================================================================================================================    
    # if "Logistic Regression".strip() in multi_function_selector:
    #     st.title("Logistic Regression")
    #     # Yêu cầu người dùng chọn biến độc lập (X) và biến phụ thuộc (y)
    #     X_columns = st.multiselect("Select the independent variable(s) (X):", options=data.columns)
    #     y_column = st.selectbox("Select the dependent variable (y):", options=data.columns)

    #     # Kiểm tra nếu người dùng đã chọn các biến
    #     if X_columns and y_column:
    #         # Nếu chỉ có một biến độc lập
    #         if len(X_columns) == 1:
    #             st.markdown(f"#### Logistic Regression for {X_columns[0]} vs {y_column}")

    #             # Lấy dữ liệu của biến độc lập (X) và biến phụ thuộc (y)
    #             X = data[X_columns]
    #             y = data[y_column]

    #             # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    #             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #             # Xây dựng mô hình logistic regression
    #             model = LogisticRegression()
    #             model.fit(X_train, y_train)

    #             # Dự đoán và đánh giá mô hình
    #             y_pred = model.predict(X_test)
    #             confusion = confusion_matrix(y_test, y_pred)
    #             report = classification_report(y_test, y_pred)
    #             st.write("Confusion Matrix:")
    #             st.write(confusion)
    #             st.write("Classification Report:")
    #             st.write(report)

    #             # Vẽ biểu đồ ROC curve
    #             y_pred_proba = model.predict_proba(X_test)[:,1]
    #             fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    #             roc_auc = auc(fpr, tpr)

    #             st.write("ROC Curve:")
    #             fig, ax = plt.subplots()
    #             ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    #             ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #             ax.set_xlim([0.0, 1.0])
    #             ax.set_ylim([0.0, 1.05])
    #             ax.set_xlabel('False Positive Rate')
    #             ax.set_ylabel('True Positive Rate')
    #             ax.set_title('Receiver Operating Characteristic')
    #             ax.legend(loc="lower right")
    #             st.pyplot(fig)
    #         # Nếu có nhiều hơn một biến độc lập
    #         elif len(X_columns) > 1:
    #             st.markdown(f"#### Logistic Regression for Multiple Independent Variables vs {y_column}")

    #             # Tạo danh sách để lưu trữ kết quả của từng biến độc lập
    #             confusion_list = []
    #             report_list = []

    #             # Lặp qua từng biến độc lập
    #             for X_column in X_columns:
    #                 st.markdown(f"##### Logistic Regression for {X_column} vs {y_column}")

    #                 # Lấy dữ liệu của biến độc lập (X) và biến phụ thuộc (y)
    #                 X = data[[X_column]]
    #                 y = data[y_column]

    #                 # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    #                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #                 # Xây dựng mô hình logistic regression
    #                 model = LogisticRegression()
    #                 model.fit(X_train, y_train)

    #                 # Dự đoán và đánh giá mô hình
    #                 y_pred = model.predict(X_test)
    #                 confusion = confusion_matrix(y_test, y_pred)
    #                 report = classification_report(y_test, y_pred)
    #                 confusion_list.append(confusion)
    #                 report_list.append(report)

    #                 # Vẽ biểu đồ ROC curve
    #                 y_pred_proba = model.predict_proba(X_test)[:,1]
    #                 fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    #                 roc_auc = auc(fpr, tpr)

    #                 st.write("ROC Curve:")
    #                 fig, ax = plt.subplots()
    #                 ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    #                 ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #                 ax.set_xlim([0.0, 1.0])
    #                 ax.set_ylim([0.0, 1.05])
    #                 ax.set_xlabel('False Positive Rate')
    #                 ax.set_ylabel('True Positive Rate')
    #                 ax.set_title('Receiver Operating Characteristic')
    #                 ax.legend(loc="lower right")
    #                 st.pyplot(fig)

    #             # Hiển thị kết quả cho từng biến độc lập
    #             for i, X_column in enumerate(X_columns):
    #                 st.write(f"Confusion Matrix for {X_column}:")
    #                 st.write(confusion_list[i])
    #                 st.write(f"Classification Report for {X_column}:")
    #                 st.write(report_list[i])
    #         # Nếu người dùng chọn nhiều biến phụ thuộc
    #         elif len(y_column) > 1:
    #             st.warning("Please select only one dependent variable (y) for logistic regression.")
    if "Logistic Regression".strip() in multi_function_selector:
        st.title("Logistic Regression")
        # Yêu cầu người dùng chọn biến độc lập (X) và biến phụ thuộc (y)
        X_columns = st.multiselect("Select the independent variable(s) (X):", options=data.columns)
        y_column = st.selectbox("Select the dependent variable (y):", options=data.columns)

        # Kiểm tra nếu người dùng đã chọn các biến
        if X_columns and y_column:
            st.markdown(f"#### Logistic Regression for Multiple Independent Variables vs {y_column}")

            # Lấy dữ liệu của biến độc lập (X) và biến phụ thuộc (y)
            X = data[X_columns]
            y = data[y_column]

            # Mã hóa biến phụ thuộc (y) nếu cần thiết
            if y.dtype == 'object' or isinstance(y.iloc[0], str):
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Mã hóa các biến độc lập (X) nếu cần thiết
            X_encoded = pd.get_dummies(X, drop_first=True)

            # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

            # Tạo subplot cho biểu đồ ROC curve
            fig, ax = plt.subplots()

            # Lặp qua từng biến độc lập đã được mã hóa
            for X_column in X_encoded.columns:
                # Xây dựng mô hình logistic regression
                model = LogisticRegression()
                model.fit(X_train[[X_column]], y_train)

                # Dự đoán và tính toán FPR và TPR
                y_pred_proba = model.predict_proba(X_test[[X_column]])[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                # Vẽ đường ROC curve cho mỗi biến độc lập
                ax.plot(fpr, tpr, lw=2, label=f'{X_column} (AUC = {roc_auc:.2f})')

            # Vẽ đường thẳng baseline
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

            # Thiết lập các thuộc tính của biểu đồ ROC curve
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")

            # Hiển thị biểu đồ ROC curve
            st.pyplot(fig)


# ====================================================================================================================   
    # if "Decision Tree".strip() in multi_function_selector:
    #     # Yêu cầu người dùng chọn biến độc lập (X) và biến phụ thuộc (y)
    #     X_column = st.selectbox("Select the independent variable (X):", options=data.columns)
    #     y_column = st.selectbox("Select the dependent variable (y):", options=data.columns)

    #     # Kiểm tra nếu người dùng đã chọn các biến
    #     if X_column and y_column:
    #         st.markdown(f"### Decision Tree for {X_column} vs {y_column}")

    #         # Lấy dữ liệu của biến độc lập (X) và biến phụ thuộc (y)
    #         X = data[[X_column]]
    #         y = data[y_column]

    #         # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #         # Xây dựng mô hình Decision Tree
    #         model = DecisionTreeClassifier()
    #         model.fit(X_train, y_train)

    #         # Dự đoán và đánh giá mô hình
    #         y_pred = model.predict(X_test)
    #         confusion = confusion_matrix(y_test, y_pred)
    #         report = classification_report(y_test, y_pred)
    #         st.write("Confusion Matrix:")
    #         st.write(confusion)
    #         st.write("Classification Report:")
    #         st.write(report)

    #         # Vẽ biểu đồ ROC curve
    #         y_pred_proba = model.predict_proba(X_test)[:,1]
    #         fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    #         roc_auc = auc(fpr, tpr)

    #         st.write("ROC Curve:")
    #         fig, ax = plt.subplots()
    #         ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    #         ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #         ax.set_xlim([0.0, 1.0])
    #         ax.set_ylim([0.0, 1.05])
    #         ax.set_xlabel('False Positive Rate')
    #         ax.set_ylabel('True Positive Rate')
    #         ax.set_title('Receiver Operating Characteristic')
    #         ax.legend(loc="lower right")
    #         st.pyplot(fig)

    #         # Chuyển đổi model.classes_ thành danh sách chuỗi
    #         class_names = [str(class_name) for class_name in model.classes_]

    #         # Vẽ cây quyết định và chuyển đổi thành chuỗi DOT
    #         dot_data = export_graphviz(model, out_file=None, 
    #                                 feature_names=X_train.columns,  
    #                                 class_names=class_names,  
    #                                 filled=True, rounded=True,  
    #                                 special_characters=True)  
    #         st.graphviz_chart(dot_data)

    if "Decision Tree".strip() in multi_function_selector:
        # Yêu cầu người dùng chọn biến độc lập (X) và biến phụ thuộc (y)
        X_columns = st.multiselect("Select the independent variable(s) (X):", options=data.columns)
        y_column = st.selectbox("Select the dependent variable (y):", options=data.columns)

        # Kiểm tra nếu người dùng đã chọn các biến
        if X_columns and y_column:
            st.markdown(f"### Decision Tree for Multiple Independent Variables vs {y_column}")

            # Lấy dữ liệu của biến độc lập (X) và biến phụ thuộc (y)
            X = data[X_columns]
            y = data[y_column]

            # Mã hóa biến phụ thuộc (y) nếu cần thiết
            if y.dtype == 'object' or isinstance(y.iloc[0], str):
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Mã hóa các biến độc lập (X) nếu cần thiết
            X_encoded = pd.get_dummies(X, drop_first=True)

            # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

            # Xây dựng mô hình Decision Tree
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)

            # Dự đoán và đánh giá mô hình
            y_pred = model.predict(X_test)
            confusion = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            st.write("Confusion Matrix:")
            st.write(confusion)
            st.write("Classification Report:")
            st.write(report)

            # Vẽ biểu đồ ROC curve
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            st.write("ROC Curve:")
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
            st.pyplot(fig)

            # Chuyển đổi model.classes_ thành danh sách chuỗi
            class_names = [str(class_name) for class_name in model.classes_]

            # Vẽ cây quyết định và chuyển đổi thành chuỗi DOT
            dot_data = export_graphviz(model, out_file=None, 
                                    feature_names=X_train.columns,  
                                    class_names=class_names,  
                                    filled=True, rounded=True,  
                                    special_characters=True)  
            st.graphviz_chart(dot_data)


# ====================================================================================================================    
    # if "KNN".strip() in multi_function_selector:
    #     # Yêu cầu người dùng chọn biến độc lập (X) và biến phụ thuộc (y)
    #     X_column = st.selectbox("Select the independent variable (X):", options=data.columns)
    #     y_column = st.selectbox("Select the dependent variable (y):", options=data.columns)

    #     # Kiểm tra nếu người dùng đã chọn các biến
    #     if X_column and y_column:
    #         st.markdown(f"### KNN for {X_column} vs {y_column}")

    #         # Lấy dữ liệu của biến độc lập (X) và biến phụ thuộc (y)
    #         X = data[[X_column]]
    #         y = data[y_column]

    #         # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #         # Xây dựng mô hình KNN
    #         model = KNeighborsClassifier()
    #         model.fit(X_train, y_train)

    #         # Dự đoán và đánh giá mô hình
    #         y_pred = model.predict(X_test)
    #         confusion = confusion_matrix(y_test, y_pred)
    #         report = classification_report(y_test, y_pred)
    #         st.write("Confusion Matrix:")
    #         st.write(confusion)
    #         st.write("Classification Report:")
    #         st.write(report)

    #         # Vẽ biểu đồ ROC curve
    #         y_pred_proba = model.predict_proba(X_test)[:,1]
    #         fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    #         roc_auc = auc(fpr, tpr)

    #         st.write("ROC Curve:")
    #         fig, ax = plt.subplots()
    #         ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    #         ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #         ax.set_xlim([0.0, 1.0])
    #         ax.set_ylim([0.0, 1.05])
    #         ax.set_xlabel('False Positive Rate')
    #         ax.set_ylabel('True Positive Rate')
    #         ax.set_title('Receiver Operating Characteristic')
    #         ax.legend(loc="lower right")
    #         st.pyplot(fig)
    
    st.set_option('deprecation.showPyplotGlobalUse', False)

    if "KNN".strip() in multi_function_selector:
        # Yêu cầu người dùng chọn các biến độc lập (X) và biến phụ thuộc (y)
        X_columns = st.multiselect("Select the independent variable(s) (X):", options=data.columns)
        y_column = st.selectbox("Select the dependent variable (y):", options=data.columns)

        # Kiểm tra nếu người dùng đã chọn các biến
        if X_columns and y_column:
            st.markdown(f"### KNN for Multiple Independent Variables vs {y_column}")

            # Lấy dữ liệu của các biến độc lập (X) và biến phụ thuộc (y)
            X = data[X_columns]
            y = data[y_column]

            # Mã hóa biến phụ thuộc (y) nếu cần thiết
            if y.dtype == 'object' or isinstance(y.iloc[0], str):
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Mã hóa các biến độc lập (X) nếu cần thiết
            X_encoded = pd.get_dummies(X, drop_first=True)

            # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

            # Xây dựng mô hình KNN
            model = KNeighborsClassifier()
            model.fit(X_train, y_train)

            # Dự đoán và đánh giá mô hình
            y_pred = model.predict(X_test)
            confusion = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            st.write("Confusion Matrix:")
            st.write(confusion)
            st.write("Classification Report:")
            st.write(report)

            # Vẽ biểu đồ Confusion Matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion, annot=True, cmap="YlGnBu", fmt="d", cbar=False)
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title('Confusion Matrix')
            
            # Hiển thị biểu đồ Confusion Matrix
            st.pyplot(plt.gcf())

            # Vẽ biểu đồ ROC curve
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            st.write("ROC Curve:")
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
            
            # Hiển thị biểu đồ ROC Curve
            st.pyplot(fig)

# ====================================================================================================================    

    if "Handling Missing Data" in multi_function_selector:
        handling_missing_value_option = st.radio("Select What you want to do", ("Drop Null Values", "Filling in Missing Values"))

        if handling_missing_value_option == "Drop Null Values":

            drop_null_values_option = st.radio("Choose your option as suted: ", ("Drop all null value rows", "Only Drop Rows that contanines all null values"))
            droped_null_value = handling_missing_values(data, drop_null_values_option)
            st.write(droped_null_value)
            export_rename_column = download_data(droped_null_value, label="fillna_column")
        
        elif handling_missing_value_option == "Filling in Missing Values":
            
            if 'missing_dict' not in st.session_state:
                st.session_state.missing_dict = {}
            
            fillna_column_selector = st.selectbox("Please Select or Enter a column Name you want to fill the NaN Values: ", options=column_with_null_values)
            fillna_text_data = st.text_input("Enter the New Value for the {} Column NaN Value".format(fillna_column_selector), max_chars=50)

            if st.button("Draft Changes", help="when you want to fill multiple columns/single column null values so first you have to click Save Draft button this updates the data and then press Rename Columns Button."):     
                
                if fillna_column_selector in num_category:
                    try:
                        st.session_state.missing_dict[fillna_column_selector] = float(fillna_text_data)
                    except:
                        st.session_state.missing_dict[fillna_column_selector] = int(fillna_text_data)
                else:
                    st.session_state.missing_dict[fillna_column_selector] = fillna_text_data

            st.code(st.session_state.missing_dict)

            if st.button("Apply Changes", help="Takes your data and Fill NaN Values for columns as your wish."):

                fillna_column = handling_missing_values(data,handling_missing_value_option, st.session_state.missing_dict)
                st.write(fillna_column)
                export_rename_column = download_data(fillna_column, label="fillna_column")
                st.session_state.missing_dict = {}

# ==========================================================================================================================================

    if "Data Wrangling" in multi_function_selector:
        data_wrangling_option = st.radio("Choose your option as suted: ", ("Merging On Index", "Concatenating On Axis"))

        if data_wrangling_option == "Merging On Index":
            data_wrangling_merging_uploaded_file = st.file_uploader("Upload Your Second file you want to merge", type=uploaded_file.name.split(".")[1])

            if data_wrangling_merging_uploaded_file is not None:

                second_data = seconddata(data_wrangling_merging_uploaded_file, file_type=data_wrangling_merging_uploaded_file.type.split("/")[1])
                same_columns = match_elements(data, second_data)
                merge_key_selector = st.selectbox("Select A Comlumn by which you want to merge on two Dataset", options=same_columns)
                
                merge_data = data_wrangling(data, second_data, merge_key_selector, data_wrangling_option)
                st.write(merge_data)
                download_data(merge_data, label="merging_on_index")

        if data_wrangling_option == "Concatenating On Axis":

            data_wrangling_concatenating_uploaded_file = st.file_uploader("Upload Your Second file you want to Concatenate", type=uploaded_file.name.split(".")[1])

            if data_wrangling_concatenating_uploaded_file is not None:

                second_data = seconddata(data_wrangling_concatenating_uploaded_file, file_type=data_wrangling_concatenating_uploaded_file.type.split("/")[1])
                concatenating_data = data_wrangling(data, second_data, None, data_wrangling_option)
                st.write(concatenating_data)
                download_data(concatenating_data, label="concatenating_on_axis")
        
# ==========================================================================================================================================
    st.sidebar.info("Nhóm 19_LeQuangNghia_VoKhacDoai.")
    if st.sidebar.button("Clear Cache"):
        clear_image_cache()
else:
    with open('C:/Users/ADMINIS/Downloads/Data-Analysis-Web-App-master/Data-Analysis-Web-App-master/Data-Analysis-Web-App-master/samples/sample.zip', 'rb') as f:
        st.sidebar.download_button(
                label="Download Sample Data and Use It",
                data=f,
                file_name='sample_data.zip',  # Đã sửa đổi tên tệp thành 'sample_data.zip'
                help = "Download some sample data and use it to explore this web app."
            )