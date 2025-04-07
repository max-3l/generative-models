from pathlib import Path
import streamlit as st
import pandas as pd
import glob
import os
import re
import altair as alt
import plotly.express as px
import json
import base64

def find_metrics_files(log_dir, regex_pattern=""):
    """
    Finds all metrics.csv files within the specified log directory,
    optionally filtering by a regex pattern.

    Args:
        log_dir (str): The root directory to search for log files.
        regex_pattern (str, optional): A regex pattern to filter the paths. Defaults to "".

    Returns:
        list: A list of paths to the metrics.csv files that match the criteria.
    """
    search_pattern = os.path.join(log_dir, "*/testtube/*/metrics.csv")
    all_files = glob.glob(search_pattern)

    if regex_pattern:
        try:
            compiled_pattern = re.compile(regex_pattern)
            filtered_files = [f for f in all_files if compiled_pattern.search(f)]
        except re.error:
            st.error(f"Invalid regex pattern: {regex_pattern}")
            return []
        return filtered_files  # Corrected return
    else:
        return all_files

def load_data(file_paths, file_names):
    """
    Loads data from the specified CSV files into a dictionary of Pandas DataFrames.

    Args:
        file_paths (list): A list of paths to the CSV files.
        file_names (list): A list of names corresponding to the files.

    Returns:
        dict: A dictionary where keys are file names and values are Pandas DataFrames.
              Returns an empty dict if no files are loaded or if there are errors.
    """
    dataframes = {}
    for file_path, file_name in zip(file_paths, file_names):
        try:
            df = pd.concat((pd.read_csv(el) for el in sorted(list(Path(file_path).parent.parent.glob("*/*.csv")))), axis=1)
            dataframes[file_name] = df.reset_index(drop=True)  # Use the provided file_name as the key
        except Exception as e:
            st.error(f"Error reading file {file_path}: {e}")
            # Consider logging the error for debugging
    return dataframes

def plot_data(dataframes, selected_y_columns, y_column_labels, x_axis_column, chart_title, plotting_library="altair", include_in_graph=None):
    """
    Plots the selected y-axis columns against the specified x-axis column for each DataFrame,
    ensuring each file and y-column is represented as a separate line, each with its own color.

    Args:
        dataframes (dict): A dictionary of Pandas DataFrames, with file *names* as keys.
        selected_y_columns (list): A list of column names to plot on the y-axis.
        y_column_labels (list): A list of labels for the y-axis columns.
        x_axis_column (str): The column name to use for the x-axis.
        chart_title (str): Title of the chart
        plotting_library (str, optional):  "altair" or "plotly". Defaults to "altair".
        include_in_graph (dict, optional): A dictionary indicating whether to include each y-column in the graph.
                                           Keys are y_column names, values are True (include) or False (exclude).
                                           If None, all selected y-columns are included.

    Returns:
        altair.Chart or plotly.graph_objects.Figure: An Altair or Plotly chart object,
                                                     or None if there's an error.
    """
    if not dataframes:
        st.warning("No data to plot.")
        return None

    if not selected_y_columns:
        st.warning("Please select at least one y-axis column to plot.")
        return None

    if len(selected_y_columns) != len(y_column_labels):
        st.error("The number of y-axis columns must match the number of y-axis labels.")
        return None

    if x_axis_column in selected_y_columns:
        y_column_labels = [*y_column_labels]
        selected_y_columns = [*selected_y_columns]
        y_column_labels.pop(selected_y_columns.index(x_axis_column))
        selected_y_columns.pop(selected_y_columns.index(x_axis_column))

    # Combine dataframes for plotting
    combined_data = pd.DataFrame()
    for file_name, df in dataframes.items():
        if x_axis_column not in df.columns:
            st.error(f"Error: The specified x-axis column '{x_axis_column}' is not present in the data from {file_name}.")
            return None

        for y_col in selected_y_columns:
            if y_col not in df.columns:
                st.error(f"Error: The y-axis column '{y_col}' is not present in the data from {file_name}.")
                return None
        # Add a 'source' column to distinguish between dataframes, now using file_name
        df['source'] = file_name
        combined_data = pd.concat([combined_data, df], ignore_index=True)

    # Melt the dataframe to long format for easier plotting
    id_vars = [x_axis_column, 'source']  # Keep x-axis and source for melting
    value_vars = selected_y_columns
    try:
        data_melted = combined_data.melt(id_vars=id_vars, value_vars=value_vars, var_name='variable', value_name='value')
    except KeyError as e:
        st.error(f"Error preparing data for plotting: {e}.  Check that the x_axis column name is correct.")
        return None

    # Rename the 'variable' column to 'metric_label'
    data_melted = data_melted.rename(columns={'variable': 'metric_label'})

    # Map the original y_columns to the user-provided labels
    label_map = dict(zip(selected_y_columns, y_column_labels))
    data_melted['metric_label'] = data_melted['metric_label'].map(label_map)
    # Create a new column that combines 'source' and 'variable' for unique coloring
    data_melted['source_variable'] = data_melted['source'] + ' - ' + data_melted['metric_label']

    # Filter data based on include_in_graph
    if include_in_graph:
        data_melted = data_melted[data_melted['metric_label'].isin([label_map[col] for col, include in include_in_graph.items() if include and col in label_map])]

    if plotting_library == "altair":
        try:
            # Create the Altair chart
            chart = alt.Chart(data_melted).mark_line(point=True).encode(
                x=alt.X(x_axis_column, title=x_axis_column),
                y=alt.Y('value', title='Value'),
                color=alt.Color('source_variable:N', legend=alt.Legend(title="Experiment - Metric")),  # Color by combined column
                tooltip=[x_axis_column, 'metric_label', 'value', 'source']
            ).properties(
                title=chart_title
            ).interactive()
            return chart
        except Exception as e:
            st.error(f"Error creating Altair plot: {e}")
            return None
    elif plotting_library == "plotly":
        try:
            # Create the Plotly chart
            chart = px.line(
                data_melted.rename(columns={"source_variable": "Legend"}),
                x=x_axis_column,
                y="value",
                color="Legend",  # Color by combined column
                hover_data=[x_axis_column, "metric_label", "value", "source"],
                title=chart_title,
            ).update_traces(connectgaps=True).update_layout({'legend': {
                            "font": {
                                "family": "Source Code Pro, monospace",
                                "size": 12,
                                "color": "black"
                            }
                        }
                    })
            return chart
        except Exception as e:
            st.error(f"Error creating Plotly plot: {e}")
            return None
    else:
        st.error(f"Invalid plotting library: {plotting_library}.  Must be 'altair' or 'plotly'.")
        return None

def calculate_and_display_table(dataframes, selected_y_columns, y_column_labels, minimize_or_maximize, include_in_graph=None):
    """
    Calculates the minimum or maximum value for each selected y-column (metric)
    across the provided dataframes, and displays the results in a Streamlit table.

    Args:
        dataframes (dict): A dictionary of Pandas DataFrames, where keys are file names.
        selected_y_columns (list): A list of column names to analyze.
        y_column_labels (list):  A list of labels for the y-axis columns.
        minimize_or_maximize (dict): A dictionary indicating whether to find the 'min' or 'max'
                                      for each y-column.  Keys are y_column names.
        include_in_graph (dict, optional): A dictionary indicating whether to include each y-column in the graph.
                                           Keys are y_column names, values are True (include) or False (exclude).
                                           If None, all selected y-columns are included.
    """
    if not dataframes:
        st.warning("No data to process for the table.")
        return

    if not selected_y_columns:
        st.warning("No y-axis columns selected for table calculation.")
        return

    # Ensure minimize_or_maximize has all required keys
    for col in selected_y_columns:
        if col not in minimize_or_maximize:
            st.error(f"Missing preference for column '{col}'.  Please specify 'min' or 'max'.")
            return

    results = {}
    for file_name, df in dataframes.items():
        results[file_name] = {}
        for y_col in selected_y_columns:
            if y_col not in df.columns:
                st.error(f"Column '{y_col}' not found in dataframe for file '{file_name}'.")
                return  # Stop processing if a column is missing

            if minimize_or_maximize[y_col] == 'min':
                results[file_name][y_col] = df[y_col].min()
            elif minimize_or_maximize[y_col] == 'max':
                results[file_name][y_col] = df[y_col].max()
            else:
                st.error(f"Invalid preference '{minimize_or_maximize[y_col]}' for column '{y_col}'.  Use 'min' or 'max'.")
                return

    # Convert the results to a Pandas DataFrame for display
    results_df = pd.DataFrame.from_dict(results, orient="index")

    # Find the best file for each metric
    best_files = {}
    for y_label, y_col in zip(y_column_labels, selected_y_columns):
        if minimize_or_maximize[y_col] == 'min':
            best_files[y_label] = results_df.idxmin()[y_col]  # Get the *file name*
        else:
            best_files[y_label] = results_df.idxmax()[y_col]  # Get the *file name*

    results_df = results_df.rename(columns=dict(zip(selected_y_columns, y_column_labels)))

    # Apply bold formatting to the best values in the DataFrame
    def highlight_best(val, current_metric, current_config):
        for metric, name in best_files.items():
            if metric == current_metric:
                if name == "N/A (Table Only)":
                    return ''
                elif name == current_config:
                    return 'font-weight: bold; background-color: #f0f0f0;'
        return ''

    styled_df = results_df.style.apply(
        lambda x: [highlight_best(v, x.name, index) for index, v in x.items()],  # Apply col-wise
        axis=0  # Important:  Apply to each *column*
    )
    # Rename columns for display

    st.subheader("Metrics by File")
    st.dataframe(styled_df, use_container_width=True)

def export_state(state_dict):
    """
    Exports the application state to a JSON string.
    
    Args:
        state_dict (dict): Dictionary containing the application state
        
    Returns:
        str: Base64 encoded JSON string of the state
    """
    state_json = json.dumps(state_dict)
    return state_json

def import_state(imported_data):
    """
    Imports the application state from a base64 encoded JSON string.
    
    Args:
        imported_data (str): Base64 encoded JSON string of the state
        
    Returns:
        dict: Dictionary containing the imported state
    """
    try:
        # decoded_data = base64.b64decode(imported_data.encode()).decode()
        return json.loads(imported_data)
    except Exception as e:
        st.error(f"Error importing state: {e}")
        return None

def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("Metrics Plotter")

    # Initialize session state for plots if it doesn't exist
    if 'plots' not in st.session_state:
        st.session_state.plots = []
    if 'update_plots' not in st.session_state:
        st.session_state.update_plots = False

    # Add export/import buttons at the top
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export State"):
            # Create state dictionary
            state = {
                'log_dir': st.session_state.get('log_dir', 'logs'),
                'regex_filter': st.session_state.get('regex_filter', ''),
                'selected_files': st.session_state.get('selected_files', []),
                'file_names': st.session_state.get('file_names', {}),
                'plots': st.session_state.plots,
                'x_axis_column': st.session_state.get('x_axis_column', 'step'),
                'plotting_library': st.session_state.get('plotting_library', 'altair')
            }
            
            # Export state to base64 string
            exported_state = export_state(state)
            
            # Create download button
            st.download_button(
                label="Download State File",
                data=exported_state,
                file_name="metrics_plotter_state.json",
                mime="application/json"
            )

    with col2:
        uploaded_file = st.file_uploader("Import State", type=['json'])
        if uploaded_file is not None:
            try:
                imported_data = uploaded_file.read().decode()
                imported_state = import_state(imported_data)
                
                if imported_state:
                    # Restore the state
                    st.session_state.log_dir = imported_state.get('log_dir', 'logs')
                    st.session_state.regex_filter = imported_state.get('regex_filter', '')
                    st.session_state.selected_files = imported_state.get('selected_files', [])
                    st.session_state.file_names = imported_state.get('file_names', {})
                    st.session_state.plots = imported_state.get('plots', [])
                    st.session_state.x_axis_column = imported_state.get('x_axis_column', 'step')
                    st.session_state.plotting_library = imported_state.get('plotting_library', 'altair')
                    st.success("State imported successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error importing state: {e}")

    log_dir = st.text_input("Enter the root log directory:", 
                          value=st.session_state.get('log_dir', 'logs'),
                          key='log_dir')
    regex_filter = st.text_input("Filter paths with regex (optional):", 
                               value=st.session_state.get('regex_filter', ''),
                               key='regex_filter')
    metrics_files = find_metrics_files(log_dir, regex_filter)

    if not metrics_files:
        st.warning(f"No metrics.csv files found in '{log_dir}' matching the pattern '{regex_filter}'.")
        return

    # Use a dictionary to store file paths and user-provided names
    file_name_dict = {}
    selected_files = st.multiselect("Select metrics.csv files:", 
                                  metrics_files, 
                                  default=st.session_state.get('selected_files', metrics_files[-1:]),
                                  format_func=lambda el: Path(el).parts[1],
                                  key='selected_files')
    
    for file_path in selected_files:
        file_name_dict[file_path] = st.text_input(
            f"Enter name for {Path(file_path).parts[1]}:",
            value=st.session_state.get('file_names', {}).get(file_path, Path(file_path).parts[1]),
            key=f"Enter name {file_path}"
        )

    # Store file names in session state
    st.session_state.file_names = file_name_dict

    # Separate file paths and names for the load_data function
    selected_files_paths = list(file_name_dict.keys())
    selected_files_names = list(file_name_dict.values())
    dataframes = load_data(selected_files_paths, selected_files_names)

    # Get all unique columns from the loaded dataframes
    all_columns = []
    for df in dataframes.values():
        all_columns.extend(df.columns)
    unique_columns = list(set(all_columns))

    try:
        index_of_step = unique_columns.index("step")
    except ValueError:
        index_of_step = 0
    
    x_axis_column = st.selectbox(
        "Choose the x-axis column:", 
        unique_columns, 
        index=index_of_step,
        key='x_axis_column'
    )
    
    plotting_library = st.radio(
        "Plotting Library", 
        options=["altair", "plotly"], 
        index=0,
        key='plotting_library'
    )

    # Use a key for the button that is unique across reruns
    if st.button("Add Graph", key=f"add_graph_{len(st.session_state.plots)}"):
        st.session_state.plots.append({
            'y_columns': [],
            'y_column_labels': [],
            'title': f"Graph {len(st.session_state.plots) + 1}",
            'minimize_or_maximize': {},
            'plotting_library': plotting_library,
            'include_in_graph': {},
        })

    for i, plot_config in enumerate(st.session_state.plots):
        st.subheader(f"Graph {i + 1}")
        plot_config['y_columns'] = st.multiselect(f"Choose y-axis columns for Graph {i + 1}:", unique_columns,
                                                  key=f"y_columns_{i}")
        # Add a text input for the labels, using the y_columns as keys.
        plot_config['y_column_labels'] = [st.text_input(f"Label for {col}:", col, key=f"y_label_{i}_{col}") for col in
                                          plot_config['y_columns']]

        # Add include_in_graph
        plot_config['include_in_graph'] = {}
        for y_col in plot_config['y_columns']:
            plot_config['include_in_graph'][y_col] = st.checkbox(
                f"Include {y_col} in graph (Graph {i + 1}):",
                value=True,  # Default to True
                key=f"include_in_graph_{i}_{y_col}"
            )

        # Add minimize/maximize selection
        plot_config['minimize_or_maximize'] = {}
        for y_col in plot_config['y_columns']:
            plot_config['minimize_or_maximize'][y_col] = st.radio(
                f"Optimize for {y_col} (Graph {i + 1}):",
                options=['min', 'max'],
                index=0,  # Default to 'min'
                key=f"min_max_{i}_{y_col}"  # Unique key per y_col
            )
        plot_config['title'] = st.text_input(f"Enter chart title for Graph {i + 1}", plot_config['title'], key=f"title_{i}")
        plot_config['plotting_library'] = plotting_library  # Keep track of the plotting library.

    # Add a button to update all plots and tables
    if st.button("Update All Plots and Tables"):
        st.session_state.update_plots = True
        st.rerun()

    # Only show plots and tables if update_plots is True
    if st.session_state.update_plots:
        for i, plot_config in enumerate(st.session_state.plots):
            chart = plot_data(dataframes, plot_config['y_columns'], plot_config['y_column_labels'], x_axis_column,
                              plot_config['title'], plot_config['plotting_library'], plot_config['include_in_graph'])
            if chart:
                if plot_config['plotting_library'] == 'altair':
                    st.altair_chart(chart, use_container_width=True)
                elif plot_config['plotting_library'] == 'plotly':
                    st.plotly_chart(chart, use_container_width=True, config={
                        'toImageButtonOptions': {
                            'format': 'svg', # one of png, svg, jpeg, webp
                            'width': 1920,
                            'height': 1080
                        }
                    })
            # Call the function to calculate and display the table.
            calculate_and_display_table(dataframes, plot_config['y_columns'], plot_config['y_column_labels'],
                                        plot_config['minimize_or_maximize'])


if __name__ == "__main__":
    main()
