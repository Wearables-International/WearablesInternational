import os
import shutil
import pytz
import pickle
import pandas as pd
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import MinMaxScaler
import taipy.gui.builder as tgb
from taipy.gui import Gui, notify, download, State, invoke_long_callback, notify
from wearablesinternational import ReaderFactory
from wearablesinternational.Models import MLModelFactory
from wearablesinternational import convert_to_local_timezone
from wearablesinternational.Preprocessing import lowpass_filter, bandpass_filter, normalize_data, zscore_data, downsample_column
from wearablesinternational.Preprocessing import scale_to_range, impute_with_mean, impute_with_median
from wearablesinternational.Artefacts import remove_outliers
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

pd.options.mode.chained_assignment = None

# region Parameters and State

run_directory = os.path.dirname(os.path.abspath(__file__))
cache_directory = os.path.join(run_directory, "cache")
models_directory = os.path.join(run_directory, "models")
splash_image = os.path.join(run_directory, "splash.jpeg")
menu = [("reset", "Reset"), ("import", "Import"), ("process", "Pre-Process"), ("artefact", "Artefacts"), ("visual", "Visualize"), ("model", "Model"), ("export", "Export"),("group", "Group"), ("about", "Software")]
timezones = ["UTC", "Australia/Brisbane", "Australia/Sydney"]
show_about_dialog = False
render_intro = True
render_expanders = False

# data file loading and processing
show_import_dialog = False
type_selector = "Empatica E4"
selected_timezone = "UTC"
file_content = ""
data_df = {}
data_df_properties = {}
data_view = pd.DataFrame(columns=["Biomarker", "Start Time", "Frequency (Hz)", "Records"])
data_view_visible = False
table_rows_selected = []

# pre-processing option selection
show_process_dialog = False
process_selector = ""
process_action = ""
process_low = 0
process_high = 0
process_fs = 0
process_order = 0
process_option_low = "disabled"
process_option_high = "disabled"
process_option_fs = "disabled"
process_option_order = "disabled"

# artefact removal
show_artefact_dialog = False
artefact_action = ""

# options pane
option_scale = True
option_filter_dates = False
option_start_date = datetime.now(pytz.UTC)
option_end_date = datetime.now(pytz.UTC) + timedelta(days=1)
option_start_date = option_start_date.astimezone(pytz.timezone(selected_timezone))
option_end_date = option_end_date.astimezone(pytz.timezone(selected_timezone)) 
option_dates = [option_start_date, option_end_date]
option_downsample = True
option_downsample_secs = 25

# visualizations
show_visuals_options_dialog = False
show_visuals_plot_dialog = False
show_visuals_corr_dialog = False
show_visuals_box_dialog = False
visuals_selector = "All"
available_signals = []
available_plots = ["Correlation", "Box Plot", "Report"]
plot_selector = "Correlation"
plot_type = ""
plot_title = ""
plot_data_chart = []
plot_data_chart_data = None

# modelling
show_model_options_dialog = False
show_model_plot_dialog = False
model_type = ""
model_title = ""
model_data_chart = []
model_selector = ""
model_available = ""

# export
export_subject = ""
show_export_dialog = False
export_available = False

# group processing
show_group_options_dialog = False
show_group_plotoptions_dialog = False
group_content = ""
group_plot_selector = ""

# use on_init to load our models so state is available when UI is drawn
def on_init(state):
    model_factory = MLModelFactory(models_directory)
    model_available = ";".join(model_factory.ListModels())
    state.model_available = model_available

# charts
sysp_visible = False
sysp_data = None
sysp_data_chart = pd.DataFrame(columns=["Period", "Value"])
eda_visible = False
eda_data = None
eda_data_chart = pd.DataFrame(columns=["Period", "Value"])
hr_visible = False
hr_data = None
hr_data_chart = pd.DataFrame(columns=["Period", "Value"])
ibi_visible = False
ibi_data = None
ibi_data_chart = pd.DataFrame(columns=["Period", "Value"])
temp_visible = False
temp_data = None
temp_data_chart = pd.DataFrame(columns=["Period", "Value"])
bvp_visible = False
bvp_data = None
bvp_data_chart = pd.DataFrame(columns=["Period", "Value"])
acc_visible = False
acc_data = None
acc_data_chart = pd.DataFrame(columns=["Period", "Value"])
rr_visible = False
rr_data = None
rr_data_chart = pd.DataFrame(columns=["Period", "Value"])
ppg_visible = False
ppg_data = None
ppg_data_chart = pd.DataFrame(columns=["Period", "Value"])
steps_visible = False
steps_data = None
steps_data_chart = pd.DataFrame(columns=["Period", "Value"])
pr_visible = False
pr_data = None
pr_data_chart = pd.DataFrame(columns=["Period", "Value"])
prv_visible = False
prv_data = None
prv_data_chart = pd.DataFrame(columns=["Period", "Value"])
activity_visible = False
activity_data = None
activity_data_chart = pd.DataFrame(columns=["Period", "Value"])
hrv_visible = False
hrv_data = None
hrv_data_chart = pd.DataFrame(columns=["Period", "Value"])
totalsleep_visible = False
totalsleep_data = None
totalsleep_data_chart = pd.DataFrame(columns=["Period", "Value"])
deepsleep_visible = False
deepsleep_data = None
deepsleep_data_chart = pd.DataFrame(columns=["Period", "Value"])
remsleep_visible = False
remsleep_data = None
remsleep_data_chart = pd.DataFrame(columns=["Period", "Value"])
efficientsleep_visible = False
efficientsleep_data = None
efficientsleep_data_chart = pd.DataFrame(columns=["Period", "Value"])

#endregion

# region ChatGPT interface

show_report_dialog = False
report_text = ""
status = 0

def get_markdown_result(prompt):
    openai_apikey="<your open ai api key here...>"
    if len(openai_apikey) > 0:
        client = OpenAI(api_key=openai_apikey)
        response = client.chat.completions.create(model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a report writer working for a mental health clinician."},
            {"role": "user", "content": prompt}
        ],temperature=0)
        markdown_result = response.choices[0].message.content
        return markdown_result

#endregion

# region Chart Configuration

chart_layout = {
    "dragmode": "select",
    "margin": {"l":0,"r":0,"b":0,"t":0}    
}

chart_config = {
    "displayModeBar": True,
    "displaylogo":False
}

chart_properties = {
    "layout": {
    "showlegend":False,
    'yaxis': {'title': '', 'visible': True, 'showticklabels': True},
    'xaxis': {'title': '','visible': True, 'showticklabels': True}  
    }
}

plot_chart_layout = {
    "annotations": [],
    "xaxis": {
        "visible": True,
        "automargin":True,
        "title":None,
        "tickfont": {
            "size":10
        }
    },
    "yaxis": {
        "visible": True,
        "automargin":True,
        "title":None,
        "tickfont": {
            "size":10
        }
    }
}

correlation_chart_options = {
    "colorscale": "Bluered",
    "mode":"text"
}

correlation_chart_layout = {
    "annotations": [],
    "xaxis": {
        "visible": True,
        "automargin":True,
        "title":None,
        "tickfont": {
            "size":10
        }
    },
    "yaxis": {
        "visible": True,
        "automargin":True,
        "title":None,
        "tickfont": {
            "size":10
        }
    }
}

correlation_chart_config = {
    "displayModeBar": False,
    "displaylogo":False
}

box_chart_layout = {
    "xaxis": {
        "visible": True,
        "title":None
    },
    "yaxis": {
        "visible": True,
        "title":None
    }
}

# endregion

# region Actions/Events

def about_action(state):
    state.show_about_dialog = False


def hide_splash(state):
    state.splash = False


def clear_cache_directory():
    if os.path.exists(cache_directory) and os.path.isdir(cache_directory):
        for file_name in os.listdir(cache_directory):
            file_path = os.path.join(cache_directory, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                pass


def on_scale_biomarkers(state):
    refresh_biomarker_charts(state, specific_biomarker=None)


def on_filter_dates(state):
    if state.option_filter_dates == True:
        refresh_biomarker_charts(state, specific_biomarker=None)


def on_select_timezone(state, id, payload):
    state.selected_timezone = payload
    state.option_start_date = option_start_date.astimezone(pytz.timezone(payload))
    state.option_end_date = option_end_date.astimezone(pytz.timezone(payload)) 
    state.option_dates = [option_start_date, option_end_date]


def on_dates_change(state, name, value):
    if state.option_filter_dates == True:
        refresh_biomarker_charts(state, specific_biomarker=None)


def on_reset(state):
    state.data_df = {}
    state.data_df_properties = {}
    state.data_view = pd.DataFrame(columns=["Biomarker", "Start Time", "Frequency (Hz)", "Records"])
    state.data_view_visible = False
    state.sysp_visible = False
    state.eda_visible = False
    state.hr_visible = False
    state.ibi_visible = False
    state.temp_visible = False
    state.ibi_visible = False
    state.acc_visible = False
    state.rr_visible = False
    state.ppg_visible = False
    state.steps_visible = False
    state.pr_visible = False
    state.prv_visible = False
    state.activity_visible = False
    state.hrv_visible = False
    state.totalsleep_visible = False
    state.deepsleep_visible = False
    state.remsleep_visible = False
    state.efficientsleep_visible


def on_artefact_option(state, id, payload):
    pass # TODO, enable/disable parameters as required


def on_artefact_action(state, id, payload):
    if payload["args"][0]==0: # first button, OK
        if len(state.table_rows_selected) == 0:
            notify(state, "E", "No biomarker selected from table...", False, 1000)
        else:
            if state.artefact_action == "Remove Outliers":
                notify(state, "I", "Processing...", False, 1000)
                biomarker = list(state.data_df.keys())[state.table_rows_selected[0]]
                df = state.data_df[biomarker]
                df = remove_outliers(df, df.columns[-1])
                state.data_df[biomarker] = df
                refresh_biomarker_charts(state, specific_biomarker=biomarker)
                notify(state, "I", "Processing completed...", False, 500)
            if state.artefact_action == "Remove Zeroes":
                notify(state, "I", "Processing...", False, 1000)
                biomarker = list(state.data_df.keys())[state.table_rows_selected[0]]
                df = state.data_df[biomarker]
                ddf = df[df.iloc[:, -1].notna() & (df.iloc[:, -1] != 0)]
                state.data_df[biomarker] = df
                refresh_biomarker_charts(state, specific_biomarker=biomarker)
                notify(state, "I", "Processing completed...", False, 500)
            if state.artefact_action == "Impute (Mean)":
                notify(state, "I", "Processing...", False, 1000)
                biomarker = list(state.data_df.keys())[state.table_rows_selected[0]]
                df = state.data_df[biomarker]
                df = impute_with_mean(df, df.columns[-1])
                state.data_df[biomarker] = df
                refresh_biomarker_charts(state, specific_biomarker=biomarker)
                notify(state, "I", "Processing completed...", False, 500)
            if state.artefact_action == "Impute (Median)":
                notify(state, "I", "Processing...", False, 1000)
                biomarker = list(state.data_df.keys())[state.table_rows_selected[0]]
                df = state.data_df[biomarker]
                df = impute_with_median(df, df.columns[-1])
                state.data_df[biomarker] = df
                refresh_biomarker_charts(state, specific_biomarker=biomarker)
                notify(state, "I", "Processing completed...", False, 500)
    else:
        state.show_artefact_dialog = False


def on_process_option(state, id, payload):
    state.process_action = payload
    if payload == "Lowpass Filter":
        state.process_option_low = "disabled"
        state.process_option_high = "enabled"
        state.process_option_fs = "enabled"
        state.process_option_order = "enabled"
    if payload == "Bandpass Filter":
        state.process_option_low = "enabled"
        state.process_option_high = "enabled"
        state.process_option_fs = "enabled"
        state.process_option_order = "enabled"
    if payload == "Z-Score":
        state.process_option_low = "disabled"
        state.process_option_high = "disabled"
        state.process_option_fs = "disabled"
        state.process_option_order = "disabled"
    if payload == "Normalize":
        state.process_option_low = "disabled"
        state.process_option_high = "disabled"
        state.process_option_fs = "disabled"
        state.process_option_order = "disabled"


def on_process_action(state, id, payload):
    if payload["args"][0]==0: # first button, OK
        if len(state.table_rows_selected) == 0:
            notify(state, "E", "No biomarker selected from table...", False, 1000)
        else:
            if state.process_action == "Lowpass Filter":
                biomarker = list(state.data_df.keys())[state.table_rows_selected[0]]
                df = state.data_df[biomarker]
                as_list = df.iloc[:, -1].tolist() # last column always has our signal (assumption!)
                filtered = lowpass_filter(as_list, process_high, process_fs, process_order)
                state.data_df[biomarker].iloc[:, -1] = filtered
            if state.process_action == "Bandpass Filter":
                biomarker = list(state.data_df.keys())[state.table_rows_selected[0]]
                df = state.data_df[biomarker]
                as_list = df.iloc[:, -1].tolist() # last column always has our signal (assumption!)
                filtered = bandpass_filter(as_list, process_low, process_high, process_fs, process_order)
                state.data_df[biomarker].iloc[:, -1] = filtered
            if state.process_action == "Z-Score":
                biomarker = list(state.data_df.keys())[state.table_rows_selected[0]]
                df = state.data_df[biomarker]
                as_list = df.iloc[:, -1].tolist() # last column always has our signal (assumption!)
                filtered = zscore_data(as_list)
                state.data_df[biomarker].iloc[:, -1] = filtered
            if state.process_action == "Normalize":
                biomarker = list(state.data_df.keys())[state.table_rows_selected[0]]
                df = state.data_df[biomarker]
                as_list = df.iloc[:, -1].tolist() # last column always has our signal (assumption!)
                filtered = normalize_data(as_list)
                state.data_df[biomarker].iloc[:, -1] = filtered
            refresh_biomarker_charts(state, biomarker)
    state.show_process_dialog = False


def on_export_action(state, id, payload):
    # only one button so always "cancel"
    state.export_subject = ""
    state.export_available = False
    state.show_export_dialog = False


def on_export_subject(state, id, payload):
    subject_data = pickle.dumps(dict(state.data_df), fix_imports=True)
    download(state, content=subject_data, name=state.export_subject + ".subject")
    state.export_subject = ""
    state.export_available = False
    state.show_export_dialog = False


def on_export_subject_change(state, id, payload):
    state.export_available = (len(state.export_subject) > 0)


def on_visuals_option_select(state, id, payload):
    if state.visuals_selector == "All":
        state.available_plots = ["Correlation", "Box Plot", "Report"]
    else:
        state.available_plots = ["Histogram"]
    state.refresh("available_plots")


def report_action(state, id, payload):
    state.show_report_dialog = False


def visuals_options_action(state, id, payload):
    if payload["args"][0]==0: # first button, OK
        if state.plot_selector == "Report":
            openai_apikey="<your open ai api key here...>"
            if len(openai_apikey) > 0:
                embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_apikey)
                vectors_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore")
                vector_store = FAISS.load_local(vectors_directory, embeddings, allow_dangerous_deserialization=True)
                results = vector_store.similarity_search("BVP, HR, IBI, TEMP, heart rate, inter beat interval, skin temperature, blood volume pulse, depression, anxiety, acute stress, chronic stess, schizophrenia, bipolar")
                prompt_context = "\n\n".join([doc.page_content for doc in results])
                min_rows = min(df.shape[0] for df in data_df.values())
                last_columns = []
                column_names = []
                for key, df in data_df.items():
                    last_column_index = -1
                    if "group" in df.columns:
                        last_column_index = -2 # have "group" column at end
                    last_column = df.iloc[:, last_column_index]
                    if state.option_downsample == False:
                        downsampled_column = last_column
                        if len(last_column) > min_rows:
                            downsampled_column = downsample_column(df, df.columns[last_column_index], min_rows)
                        last_columns.append(downsampled_column.reset_index(drop=True))
                    else:
                        last_columns.append(last_column)
                    column_names.append(key)
                combined_df = pd.concat(last_columns, axis=1)
                combined_df.columns = column_names
                combined_df.dropna(axis=0, inplace=True)
                statistics = combined_df.describe()
                mean_EDA = statistics.loc["mean", "EDA"]
                std_EDA = statistics.loc["std", "EDA"]
                min_EDA = statistics.loc["std", "EDA"]
                max_EDA = statistics.loc["std", "EDA"]
                mean_HR = statistics.loc["mean", "HR"]
                std_HR = statistics.loc["std", "HR"]
                min_HR = statistics.loc["std", "HR"]
                max_HR = statistics.loc["std", "HR"]
                mean_TEMP = statistics.loc["mean", "TEMP"]
                std_TEMP = statistics.loc["std", "TEMP"]
                min_TEMP = statistics.loc["std", "TEMP"]
                max_TEMP = statistics.loc["std", "TEMP"]
                mean_BVP = statistics.loc["mean", "BVP"]
                std_BVP = statistics.loc["std", "BVP"]
                min_BVP = statistics.loc["std", "BVP"]
                max_BVP = statistics.loc["std", "BVP"]
                mean_IBI = statistics.loc["mean", "IBI"]
                std_IBI = statistics.loc["std", "IBI"]
                min_IBI = statistics.loc["std", "IBI"]
                max_IBI = statistics.loc["std", "IBI"]
                prompt = f"""
                You are a report writer working for a mental health clinician. You are to write a report titled Biomarker Analysis Report based on scientific journal findings regarding how biomarkers are expressed in various mental health conditions, including healthy individuals, acute stress, chronic stress, anxiety, depression, bipolar disorder, and schizophrenia. 

                The following biomarkers are provided with their mean, standard deviation, minimum, and maximum values:

                - **Electrodermal Activity (EDA):**
                - Mean: {mean_EDA}
                - Standard Deviation: {std_EDA}
                - Minimum: {min_EDA}
                - Maximum: {max_EDA}

                - **Heart Rate (HR):**
                - Mean: {mean_HR}
                - Standard Deviation: {std_HR}
                - Minimum: {min_HR}
                - Maximum: {max_HR}

                - **Skin Temperature (TEMP):**
                - Mean: {mean_TEMP}
                - Standard Deviation: {std_TEMP}
                - Minimum: {min_TEMP}
                - Maximum: {max_TEMP}

                - **Blood Volume Pulse (BVP):**
                - Mean: {mean_BVP}
                - Standard Deviation: {std_BVP}
                - Minimum: {min_BVP}
                - Maximum: {max_BVP}

                - **Interbeat Interval (IBI):**
                - Mean: {mean_IBI}
                - Standard Deviation: {std_IBI}
                - Minimum: {min_IBI}
                - Maximum: {max_IBI}

                Write a report in markdown format to indicate any significant findings between these biomarker values and the mentioned mental health conditions, based on findings from scientific literature. Do not include relevant scientific literature references. Directly link any scientific literature findings to the provided biomarker values to reach a conlusion on any potential diagnosis or correlation with known mental health conditions. Use the following context for relevance: {prompt_context}"
                """
                notify(state, "I", "Report generation in progress, please wait...", False, 5000)
                invoke_long_callback(state, get_markdown_result, [str(prompt)], heavy_status, [], 1000)
        if state.plot_selector == "Histogram":
            state.plot_title = "Histogram of " + state.visuals_selector
            state.plot_type = "histogram"
            df = state.data_df[state.visuals_selector]
            state.plot_data_chart = df.iloc[:, -1].tolist() # last column always has our signal (assumption!)
            state.show_visuals_plot_dialog = True
        if state.plot_selector == "Box Plot":
            min_rows = min(df.shape[0] for df in data_df.values())
            last_columns = []
            column_names = []
            for key, df in data_df.items():
                last_column_index = -1
                if "group" in df.columns:
                    last_column_index = -2 # have "group" column at end
                last_column = df.iloc[:, last_column_index]
                if state.option_downsample == False:
                    downsampled_column = last_column
                    if len(last_column) > min_rows:
                        downsampled_column = downsample_column(df, df.columns[last_column_index], min_rows)
                    last_columns.append(downsampled_column.reset_index(drop=True))
                else:
                    last_columns.append(last_column)
                column_names.append(key)
            combined_df = pd.concat(last_columns, axis=1)
            combined_df.columns = column_names
            combined_df.dropna(axis=0, inplace=True)
            melted_df = combined_df.melt(var_name="Columns", value_name="Values")
            state.plot_data_chart = melted_df
            state.show_visuals_box_dialog = True
        if state.plot_selector == "Correlation":
            state.plot_chart_layout = {
                "annotations": [],
                "xaxis": {
                    "visible": True,
                    "automargin":True,
                    "title":None,
                    "tickfont": {
                        "size":10
                    }
                },
                "yaxis": {
                    "visible": True,
                    "automargin":True,
                    "title":None,
                    "tickfont": {
                        "size":10
                    }
                }
            }
            min_rows = min(df.shape[0] for df in data_df.values())
            last_columns = []
            column_names = []
            for key, df in data_df.items():
                last_column_index = -1
                if "group" in df.columns:
                    last_column_index = -2 # have "group" column at end
                last_column = df.iloc[:, last_column_index]
                if state.option_downsample == False:
                    downsampled_column = last_column
                    if len(last_column) > min_rows:
                        downsampled_column = downsample_column(df, df.columns[last_column_index], min_rows)
                    last_columns.append(downsampled_column.reset_index(drop=True))
                else:
                    last_columns.append(last_column)
                column_names.append(key)
            combined_df = pd.concat(last_columns, axis=1)
            combined_df.columns = column_names
            df_corr = combined_df.corr()
            x = df_corr.columns.tolist()
            y = df_corr.index.tolist()
            z = df_corr.values.tolist()
            for i, xx in enumerate(x):
                for j, yy in enumerate(y):
                    annotation= dict(
                            x=xx,
                            y=yy,
                            text=str(round(z[j][i], 2)),
                            showarrow=False,
                            font=dict(size=10, color="white"),
                            align="center",
                            xref="x",
                            yref="y",
                        )

                    state.plot_chart_layout["annotations"].append(annotation)
            state.plot_data_chart = {"x": x, "y": y, "z": z}
            state.show_visuals_corr_dialog = True
    state.show_visuals_options_dialog = False


def heavy_status(state: State, status, markdown: str):
    if isinstance(status, bool):
        if status:
            state.report_text = markdown
            state.refresh("report_text")
            state.show_report_dialog = True
    else:
        state.status += 1

def visuals_plot_action(state):
    state.show_visuals_plot_dialog = False
    state.show_visuals_corr_dialog = False
    state.show_visuals_box_dialog = False


def model_plot_action(state):
    state.show_model_plot_dialog = False


def on_model_option_select(state, id, payload):
    have_HR = False
    have_EDA = False
    for biomarker in state.data_df.keys():
        if biomarker == "HR":
            have_HR = True
        if biomarker == "EDA":
            have_EDA = True
    if not (have_EDA and have_HR):
        notify(state, "E", "Both HR and EDA biomarkers are required for modelling...", False, 2000)
        state.model_selector = ""


def model_options_action(state, id, payload):
    if payload["args"][0]==0: # first button, OK
        state.plot_chart_layout = {
            "annotations": [],
            "xaxis": {
                "visible": True,
                "automargin":True,
                "title":None,
                "tickfont": {
                    "size":10
                }
            },
            "yaxis": {
                "visible": True,
                "automargin":True,
                "title":None,
                "tickfont": {
                    "size":10
                }
            }
        }
        hr = state.data_df["HR"]
        eda = state.data_df["EDA"]
        hr = hr.drop(columns=["timestamp_unix"])
        eda = eda.drop(columns=["timestamp_unix"])
        df_merged = pd.merge(hr, eda, on="timestamp_iso")
        model_factory = MLModelFactory(models_directory)
        model = model_factory.GetModel(state.model_selector)
        hr = df_merged["HR"].tolist()
        eda = df_merged["EDA"].tolist()
        temp = pd.DataFrame({"HR": hr,"EDA": eda})
        df_merged["Model"] = model.predict(temp[["HR","EDA"]])
        df_merged.rename(columns={'timestamp_iso': 'Date'}, inplace=True)
        scaler = MinMaxScaler()
        columns_to_scale = ['HR', 'EDA', 'Model']
        df_merged[columns_to_scale] = scaler.fit_transform(df_merged[columns_to_scale])
        state.model_data_chart = df_merged
        state.model_type = state.model_selector
        state.model_title = state.model_selector + " Model"
        state.show_model_plot_dialog = True
    state.show_model_options_dialog = False


def on_table_selection(state, var_name, action, payload):
    state.table_rows_selected = [action["index"]]
    state.refresh(var_name)


def refresh_biomarker_charts(state, specific_biomarker=None):
    if len(state.data_df) > 0:
        temp_signals = ["All"]
        for biomarker in state.data_df.keys():
            notify(state, "I", "Processing." + biomarker + "..", False, 1000)
            if (specific_biomarker == None) or (biomarker == specific_biomarker):
                if biomarker == "SYSP":
                    temp_signals.append("SYSP")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["SYSP"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.sysp_data_chart = {"Period": timestamps, "Value": rate}
                    state.sysp_visible = True
                if biomarker == "EDA":
                    temp_signals.append("EDA")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["EDA"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.eda_data_chart = {"Period": timestamps, "Value": rate}
                    state.eda_visible = True
                if biomarker == "HR":
                    temp_signals.append("HR")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["HR"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.hr_data_chart = {"Period": timestamps, "Value": rate}
                    state.hr_visible = True
                if biomarker == "IBI":
                    temp_signals.append("IBI")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["IBI"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.ibi_data_chart = {"Period": timestamps, "Value": rate}
                    state.ibi_visible = True
                if biomarker == "BVP":
                    temp_signals.append("BVP")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["BVP"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.bvp_data_chart = {"Period": timestamps, "Value": rate}
                    state.bvp_visible = True
                if biomarker == "ACC":
                    temp_signals.append("ACC")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["ACC"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.acc_data_chart = {"Period": timestamps, "Value": rate}
                    state.acc_visible = True
                if biomarker == "TEMP":
                    temp_signals.append("TEMP")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["TEMP"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.temp_data_chart = {"Period": timestamps, "Value": rate}
                    state.temp_visible = True
                if biomarker == "STEPS":
                    temp_signals.append("STEPS")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["STEPS"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.steps_data_chart = {"Period": timestamps, "Value": rate}
                    state.steps_visible = True
                if biomarker == "PPG":
                    temp_signals.append("PPG")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["PPG"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.ppg_data_chart = {"Period": timestamps, "Value": rate}
                    state.ppg_visible = True
                if biomarker == "RR":
                    temp_signals.append("RR")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["RR"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.rr_data_chart = {"Period": timestamps, "Value": rate}
                    state.rr_visible = True
                if biomarker == "PR":
                    temp_signals.append("PR")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["PR"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.pr_data_chart = {"Period": timestamps, "Value": rate}
                    state.pr_visible = True
                if biomarker == "PRV":
                    temp_signals.append("PRV")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["PRV"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.prv_data_chart = {"Period": timestamps, "Value": rate}
                    state.prv_visible = True
                if biomarker == "ACTIVITY":
                    temp_signals.append("ACTIVITY")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["ACTIVITY"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.activity_data_chart = {"Period": timestamps, "Value": rate}
                    state.activity_visible = True
                if biomarker == "HRV":
                    temp_signals.append("HRV")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["HRV"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.hrv_data_chart = {"Period": timestamps, "Value": rate}
                    state.hrv_visible = True
                if biomarker == "TOTALSLEEP":
                    temp_signals.append("TOTALSLEEP")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["TOTALSLEEP"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.totalsleep_data_chart = {"Period": timestamps, "Value": rate}
                    state.totalsleep_visible = True
                if biomarker == "REMSLEEP":
                    temp_signals.append("REMSLEEP")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["REMSLEEP"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.remsleep_data_chart = {"Period": timestamps, "Value": rate}
                    state.remsleep_visible = True
                if biomarker == "DEEPSLEEP":
                    temp_signals.append("DEEPSLEEP")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["DEEPSLEEP"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.deepsleep_data_chart = {"Period": timestamps, "Value": rate}
                    state.deepsleep_visible = True
                if biomarker == "EFFICIENTSLEEP":
                    temp_signals.append("EFFICIENTSLEEP")
                    df = state.data_df[biomarker]
                    if state.option_filter_dates == True:
                        df = df[(pd.to_datetime(df["timestamp_iso"]) >= state.option_dates[0]) & (pd.to_datetime(df["timestamp_iso"]) <= state.option_dates[1])]
                    if state.option_downsample == True and state.option_downsample_secs > 0:
                        df['group'] = df['timestamp_unix'] // state.option_downsample_secs
                        df = df.groupby('group').agg({'timestamp_unix': 'first','timestamp_iso': 'first',biomarker: 'mean'}).reset_index(drop=True)
                    timestamps = df["timestamp_iso"].tolist()
                    timestamps = convert_to_local_timezone(timestamps, selected_timezone)
                    rate = df["EFFICIENTSLEEP"].to_numpy()
                    if state.option_scale == True:
                        rate = scale_to_range(rate)
                    state.efficientsleep_data_chart = {"Period": timestamps, "Value": rate}
                    state.efficientsleep_visible = True

        state.available_signals = temp_signals
    else:
        state.available_signals = ""


def import_action(state, id, payload):
    if payload["args"][0]==0: # first button, OK
        reader = None
        if state.type_selector == "Empatica E4":
            reader = ReaderFactory.GetReader("E4")
        if state.type_selector == "Empatica Embrace Plus":
            reader = ReaderFactory.GetReader("EmbracePlus")
        if state.type_selector == "NOWATCH":
            reader = ReaderFactory.GetReader("NoWatch")
        if state.type_selector == "Oura Ring":
            reader = ReaderFactory.GetReader("OuraRing")
        temp_df = pd.DataFrame(columns=["Biomarker", "Start Time", "Frequency (Hz)", "Records"])
        if len(os.listdir(os.path.join(cache_directory))) > 0:
            notify(state, "I", "Processing...", False, 2000)
            new_datasets = reader.read(cache_directory)
            if len(state.data_df) == 0:
                for biomarker in new_datasets.keys():
                    state.data_df[biomarker] = new_datasets[biomarker].dataframe
                    state.data_df[biomarker].sort_values(by=["timestamp_iso"], ascending=True)
                    state.data_df_properties[biomarker] = [datetime.fromtimestamp(new_datasets[biomarker].start_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), new_datasets[biomarker].frequency_hz, new_datasets[biomarker].record_count]
                    row = [biomarker, state.data_df_properties[biomarker][0], state.data_df_properties[biomarker][1], state.data_df_properties[biomarker][2]]
                    temp_df.loc[len(temp_df)] = row
            else:
                temp_df = state.data_view
                for biomarker in new_datasets.keys():
                    if biomarker in state.data_df.keys():
                        temp = state.data_df[biomarker]
                        new_temp = new_datasets[biomarker].dataframe
                        state.data_df[biomarker] = pd.concat([temp, new_temp], axis=0, ignore_index=True)
                        state.data_df[biomarker] = state.data_df[biomarker].drop_duplicates()
                        state.data_df[biomarker].sort_values(by=["timestamp_iso"], ascending=True)
                        start_time = datetime.fromisoformat(state.data_df[biomarker].loc[0, "timestamp_iso"]).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                        state.data_df_properties[biomarker] = [start_time, new_datasets[biomarker].frequency_hz, len(state.data_df[biomarker])]
                        row = [biomarker, state.data_df_properties[biomarker][0], state.data_df_properties[biomarker][1], state.data_df_properties[biomarker][2]]
                        temp_df[temp_df["Biomarker"] == biomarker] = pd.Series(row)
                    else:
                        state.data_df[biomarker] = new_datasets[biomarker].dataframe
                        state.data_df[biomarker].sort_values(by=["timestamp_iso"], ascending=True)
                        state.data_df_properties[biomarker] = [datetime.fromtimestamp(new_datasets[biomarker].start_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), new_datasets[biomarker].frequency_hz, new_datasets[biomarker].record_count]
                        row = [biomarker, state.data_df_properties[biomarker][0], state.data_df_properties[biomarker][1], state.data_df_properties[biomarker][2]]
                        temp_df.loc[len(temp_df)] = row
        if len(state.data_df) > 0:
            state.data_view = temp_df
            state.data_view_visible = True
            state.render_intro = False
            state.render_expanders = True
        else:
            state.data_view_visible = False
            state.render_intro = True
            state.render_expanders = False
        clear_cache_directory()
        refresh_biomarker_charts(state, specific_biomarker=None)
    state.show_import_dialog = False


def on_upload_group_files(state):
    clear_cache_directory()
    if type(state.group_content) is str:
        files = state.group_content.split(";")
    else:
        files = state.group_content
    for file in files:
        shutil.move(file, os.path.join(cache_directory, os.path.basename(file))) 
    temp = []
    for file in os.listdir(cache_directory):
        if ".subject" in file:
            temp.append(os.path.basename(file).replace(".subject",""))
    state.group_content = ""


def group_action(state, id, payload):
    state.group_content = ""
    state.show_group_options_dialog = False
    if payload["args"][0]==0: # first button, OK
        state.show_group_plotoptions_dialog = True
    else: # cancel
        clear_cache_directory()


def group_plotoptions_action(state, id, payload):
    if payload["args"][0]==0: # first button, OK
        all_dfs = []
        for file in os.listdir(cache_directory):
            if ".subject" in file:
                with open(os.path.join(cache_directory, file) , 'rb') as f:
                    df_loaded = pickle.load(f)
                    last_columns = []
                    column_names = []
                    min_rows = min(df.shape[0] for df in df_loaded.values())
                    for _, df_main in df_loaded.items():
                        last_column_index = -1
                        if "group" in df_main.columns:
                            last_column_index = -2 # have "group" column at end
                        last_column = df_main.iloc[:, last_column_index]
                        last_column_name = df_main.columns[last_column_index]
                        downsampled_column = last_column
                        if state.option_downsample == False:
                            if len(last_column) > min_rows:
                                downsampled_column = downsample_column(df_main, df_main.columns[last_column_index], min_rows)
                        last_columns.append(downsampled_column.reset_index(drop=True))
                        column_names.append(last_column_name)
                    combined_df = pd.concat(last_columns, axis=1)
                    combined_df.columns = column_names
                    all_dfs.append(combined_df)
                    f.close()
        combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)
        if state.group_plot_selector == "Box Plot":
            clear_cache_directory()
            melted_df = combined_df.melt(var_name="Columns", value_name="Values")
            melted_df.dropna(axis=0, inplace=True)
            state.plot_data_chart = melted_df
            state.show_visuals_box_dialog = True
            state.show_group_plotoptions_dialog = False
        if state.group_plot_selector == "Correlation":
            state.plot_chart_layout = {
                "annotations": [],
                "xaxis": {
                    "visible": True,
                    "automargin":True,
                    "title":None,
                    "tickfont": {
                        "size":10
                    }
                },
                "yaxis": {
                    "visible": True,
                    "automargin":True,
                    "title":None,
                    "tickfont": {
                        "size":10
                    }
                }
            }
            df_corr = combined_df.corr()
            x = df_corr.columns.tolist()
            y = df_corr.index.tolist()
            z = df_corr.values.tolist()
            for i, xx in enumerate(x):
                for j, yy in enumerate(y):
                    annotation= dict(
                            x=xx,
                            y=yy,
                            text=str(round(z[j][i], 2)),
                            showarrow=False,
                            font=dict(size=10, color="white"),
                            align="center",
                            xref="x",
                            yref="y",
                        )

                    state.plot_chart_layout["annotations"].append(annotation)
            state.plot_data_chart = {"x": x, "y": y, "z": z}
            clear_cache_directory()
            state.show_group_plotoptions_dialog = False
            state.show_visuals_corr_dialog = True
    if payload["args"][0]==1: # Back
        state.group_content = ""
        clear_cache_directory()
        state.show_group_plotoptions_dialog = False
        state.show_group_options_dialog = True
    if payload["args"][0]==2: # Cancel
        state.show_group_plotoptions_dialog = False
        clear_cache_directory()

def on_upload_files(state):
    clear_cache_directory()
    if type(state.file_content) is str:
        files = state.file_content.split(";")
    else:
        files = state.file_content
    for file in files:
        shutil.move(file, os.path.join(cache_directory, os.path.basename(file))) 


def menu_action(state, id, payload):
    if payload["args"][0] == "import":
        state.show_import_dialog = True
    if payload["args"][0] == "reset":
        on_reset(state)
    if payload["args"][0] == "process" and len(data_df) > 0:
        state.show_process_dialog = True
    if payload["args"][0] == "artefact" and len(data_df) > 0:
        state.show_artefact_dialog = True
    if payload["args"][0] == "visual" and len(data_df) > 0:
        state.show_visuals_options_dialog = True
    if payload["args"][0] == "model" and len(data_df) > 0:
        state.show_model_options_dialog = True
    if payload["args"][0] == "about":
        state.show_about_dialog = True
    if payload["args"][0] == "export" and len(data_df) > 0:
        state.show_export_dialog = True
    if payload["args"][0] == "group":
        state.show_group_options_dialog = True        

# endregion

# region UI

with tgb.Page() as import_dialog:
    with tgb.layout():
        with tgb.part():
            tgb.selector(value="{type_selector}", mode="radio", lov="Empatica E4;Empatica Embrace Plus;NOWATCH;Oura Ring", multiple=False)
            tgb.file_selector("{file_content}", label="Select Files", notify=False, on_action=on_upload_files, extensions=".csv, .txt, .avro", drop_message="Drop To Process", multiple=True)


with tgb.Page() as process_dialog:
    with tgb.layout():
        with tgb.part():
            tgb.selector(value="{process_selector}", mode="radio", lov="Lowpass Filter;Bandpass Filter;Z-Score;Normalize", on_change=on_process_option, multiple=False)
            tgb.number("{process_low}", label="Low", min=0, max=60, class_name = "{process_option_low}")
            tgb.number("{process_high}", label="High", min=0, max=60, class_name="{process_option_high}")
            tgb.number("{process_fs}", label="FS", min=0, max=60, class_name="{process_option_fs}")
            tgb.number("{process_order}", label="Order", min=0, max=60, class_name="{process_option_order}")


with tgb.Page() as artefact_dialog:
    with tgb.layout():
        with tgb.part():
            tgb.selector(value="{artefact_action}", mode="radio", lov="Remove Outliers;Remove Zeroes; Impute (Mean); Impute (Median)", on_change=on_artefact_option, multiple=False)


with tgb.Page() as visuals_options_dialog:
    with tgb.layout():
        with tgb.part():
            tgb.selector(value="{visuals_selector}", mode="radio", lov="{available_signals}", on_change=on_visuals_option_select, multiple=False)
            tgb.text("---", mode="markdown")
            tgb.selector(value="{plot_selector}", mode="radio", lov="{available_plots}",  multiple=False)


with tgb.Page() as visuals_plot_dialog:
    tgb.chart("{plot_data_chart}", height="fit-content", width="fit-content", rebuild=True, title="{plot_title}", layout="{plot_chart_layout}")


with tgb.Page() as visuals_box_dialog:
    tgb.chart("{plot_data_chart}", type="box", height="fit-content", x="Columns", y="Values", width="fit-content", rebuild=True, title="Box Plot", layout="{box_chart_layout}")


with tgb.Page() as visuals_corr_dialog:
    tgb.chart("{plot_data_chart}", type="heatmap", height="fit-content", width="fit-content", title="Correlation", x="x", y="y", z="z", rebuild=True, options="{correlation_chart_options}", layout="{correlation_chart_layout}")


with tgb.Page() as model_options_dialog:
    with tgb.layout():
        with tgb.part():
            tgb.selector(value="{model_selector}", mode="radio", lov="Arousal;Attention;Stress;Valence", on_change=on_model_option_select, multiple=False)


with tgb.Page() as model_plot_dialog:
    tgb.chart("{model_data_chart}", height="fit-content", width="fit-content", mode="lines", x="Date", y__1="HR", y__2="EDA", y__3="Model", color__1="blue", color__2="green", color__3="red", rebuild=True, title="{model_title}", layout="{plot_chart_layout}")


with tgb.Page() as export_options_dialog:
    with tgb.layout():
        with tgb.part():
            tgb.input(value="{export_subject}", label="Subject ID", lines_shown=1, on_change=on_export_subject_change)
            tgb.file_download("{None}", on_action=on_export_subject, render="{export_available}")


with tgb.Page() as group_options_dialog:
    with tgb.layout():
        with tgb.part():
            tgb.file_selector("{group_content}", label="Select Subject Files", notify=False, on_action=on_upload_group_files, extensions=".subject", drop_message="Drop Subjects To Select", multiple=True)


with tgb.Page() as group_plotoptions_dialog:
    with tgb.layout():
        with tgb.part():
            tgb.selector(value="{group_plot_selector}", mode="radio", lov="Correlation;Box Plot", multiple=False)


with tgb.Page() as about_dialog:
    tgb.text(value="Workbench Version 0.1")
    tgb.image("{splash_image}")
    tgb.text(value="https://github.com/PCdLf/wearables_international")
    tgb.text(value="Created By: Peter de Looff, Selin Acan, Gideon Vos")


with tgb.Page() as report_dialog:
    with tgb.layout():
        with tgb.part():
            tgb.text(value="{report_text}", mode="markdown")


with tgb.Page() as page_main:
    with tgb.layout("0.2 1 0.4"):
        with tgb.part():
            tgb.menu(label = "Menu", lov = menu, on_action = menu_action)
        with tgb.part():
            with tgb.layout("1"):
                with tgb.part(render="{render_intro}"):
                    tgb.text("## Welcome to the Data Analysis Workbench for Wearable Devices.\n\n To start, load data files by clicking on the **I** (Import) button to the left. System-wide configuration options can be configured by using the **Options** pane to the right.", mode="markdown", class_name="center_text")
                with tgb.part(render="{render_expanders}"):
                    with tgb.expandable(title="Data", expanded="{data_view_visible}"):
                        tgb.table("{data_view}", editable=False, rebuild=True, downloadable=True, active=True, width="100%", on_action=on_table_selection, selected="{table_rows_selected}")
                    with tgb.expandable(title="Biomarkers", expanded="{data_view_visible}"):
                        with tgb.part(render="{eda_visible}"):
                            with tgb.expandable(title="Electrodermal Activity", expanded=True):
                                tgb.chart("{eda_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
                        with tgb.part(render="{hr_visible}"):
                            with tgb.expandable(title="Heart Rate", expanded=True):
                                tgb.chart("{hr_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
                        with tgb.part(render="{sysp_visible}"):
                            with tgb.expandable(title="Systolic Blood Pressure", expanded=True):
                                tgb.chart("{sysp_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
                        with tgb.part(render="{bvp_visible}"):
                            with tgb.expandable(title="Blood Volume Pulse", expanded=True):
                                tgb.chart("{bvp_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
                        with tgb.part(render="{ibi_visible}"):
                            with tgb.expandable(title="Interbeat Interval", expanded=True):
                                tgb.chart("{ibi_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
                        with tgb.part(render="{temp_visible}"):
                            with tgb.expandable(title="Temperature", expanded=True):
                                tgb.chart("{temp_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
                        with tgb.part(render="{acc_visible}"):
                            with tgb.expandable(title="Accelerometer", expanded=True):
                                tgb.chart("{acc_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
                        with tgb.part(render="{rr_visible}"):
                            with tgb.expandable(title="Respitory Rate", expanded=True):
                                tgb.chart("{rr_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
                        with tgb.part(render="{ppg_visible}"):
                            with tgb.expandable(title="Photoplethysmography", expanded=True):
                                tgb.chart("{ppg_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")              
                        with tgb.part(render="{steps_visible}"):
                            with tgb.expandable(title="Steps", expanded=True):
                                tgb.chart("{steps_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")              
                        with tgb.part(render="{pr_visible}"):
                            with tgb.expandable(title="Pulse Rate", expanded=True):
                                tgb.chart("{pr_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
                        with tgb.part(render="{prv_visible}"):
                            with tgb.expandable(title="Pulse Rate Variation", expanded=True):
                                tgb.chart("{prv_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}") 
                        with tgb.part(render="{activity_visible}"):
                            with tgb.expandable(title="Activity", expanded=True):
                                tgb.chart("{activity_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")
                        with tgb.part(render="{hrv_visible}"):
                            with tgb.expandable(title="Heart Rate Variation", expanded=True):
                                tgb.chart("{hrv_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}") 
                        with tgb.part(render="{totalsleep_visible}"):
                            with tgb.expandable(title="Total Sleep", expanded=True):
                                tgb.chart("{totalsleep_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")     
                        with tgb.part(render="{remsleep_visible}"):
                            with tgb.expandable(title="REM Sleep", expanded=True):
                                tgb.chart("{remsleep_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")      
                        with tgb.part(render="{deepsleep_visible}"):
                            with tgb.expandable(title="Deep Sleep", expanded=True):
                                tgb.chart("{deepsleep_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}")   
                        with tgb.part(render="{efficientsleep_visible}"):
                            with tgb.expandable(title="Efficient Sleep", expanded=True):
                                tgb.chart("{efficientsleep_data_chart}", mode="markers", height="300px", x="Period", y="Value", rebuild=True, layout="{chart_layout}", plot_config="{chart_config}", properties="{chart_properties}") 
        with tgb.part():
            with tgb.expandable(title="Options", expanded=False):
                tgb.toggle(value="{option_scale}", label="Scale Biomarker Values", on_change=on_scale_biomarkers)
                tgb.toggle(value="{option_filter_dates}", label="Filter Data by Date", on_change=on_filter_dates)
                tgb.date_range("{option_dates}", label="Range Filter", on_change=on_dates_change, with_time=True)
                tgb.text("", mode="pre")
                tgb.selector(value="{selected_timezone}", label="Timezone", lov=timezones, on_change = on_select_timezone, dropdown=True)
                tgb.toggle(value="{option_downsample}", label="Downsample Biomarker Values")
                tgb.number(value="{option_downsample_secs}", label="Seconds", min=0, max=60)


    tgb.dialog("{show_import_dialog}", title="Import Data", page="import_dialog", on_action=import_action, labels="OK;Cancel")
    tgb.dialog("{show_process_dialog}", title="Pre-Process Data", page="process_dialog", on_action=on_process_action, labels="OK;Cancel")
    tgb.dialog("{show_artefact_dialog}", title="Artefacts", page="artefact_dialog", on_action=on_artefact_action, labels="OK;Cancel")
    tgb.dialog("{show_visuals_options_dialog}", title="Visualize", page="visuals_options_dialog", on_action=visuals_options_action, labels="OK;Cancel")
    tgb.dialog("{show_visuals_plot_dialog}", title="Visualize", width="fit-content", height="60%", page="visuals_plot_dialog", on_action=visuals_plot_action, labels="OK")
    tgb.dialog("{show_visuals_corr_dialog}", title="Visualize", width="fit-content", height="60%", page="visuals_corr_dialog", on_action=visuals_plot_action, labels="OK")
    tgb.dialog("{show_visuals_box_dialog}", title="Visualize", width="fit-content", height="fit-content", page="visuals_box_dialog", on_action=visuals_plot_action, labels="OK")
    tgb.dialog("{show_model_options_dialog}", title="Modelling", page="model_options_dialog", on_action=model_options_action, labels="OK; Cancel")
    tgb.dialog("{show_model_plot_dialog}", width="fit-content", height="60%", title="Modelling", page="model_plot_dialog", on_action=model_plot_action, labels="OK")
    tgb.dialog("{show_about_dialog}", width="fit-content", class_name="about", title="About", page="about_dialog", on_action=about_action, labels="OK")
    tgb.dialog("{show_export_dialog}", title="Export Subject Data", page="export_options_dialog", on_action=on_export_action, labels="Cancel")
    tgb.dialog("{show_group_options_dialog}", title="Load Group Subjects", page="group_options_dialog", on_action=group_action, labels="OK;Cancel")
    tgb.dialog("{show_group_plotoptions_dialog}", title="Group Visualization", page="group_plotoptions_dialog", on_action=group_plotoptions_action, labels="OK;Back;Cancel")
    tgb.dialog("{show_report_dialog}", title="Subject Report", width="fit-content", height="fit-content", page="report_dialog", on_action=report_action, labels="OK")

if __name__ == "__main__":
    pages = {"page_main": page_main, "import_dialog": import_dialog, "process_dialog": process_dialog, \
             "visuals_options_dialog": visuals_options_dialog, "visuals_plot_dialog":visuals_plot_dialog, \
             "model_options_dialog": model_options_dialog, "model_plot_dialog":model_plot_dialog, \
             "visuals_corr_dialog": visuals_corr_dialog, "about_dialog": about_dialog, \
             "artefact_dialog":artefact_dialog, "visuals_box_dialog": visuals_box_dialog, \
             "export_options_dialog": export_options_dialog, "group_options_dialog": group_options_dialog, \
             "group_plotoptions_dialog":group_plotoptions_dialog, "report_dialog": report_dialog}
    gui = Gui(pages=pages).run(title="Wearables International Data Analysis Workbench", dark_mode=False)

#endregion