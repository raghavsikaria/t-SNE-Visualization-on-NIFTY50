##########################################################
# Author: Raghav Sikaria
# LinkedIn: https://www.linkedin.com/in/raghavsikaria/
# Github: https://github.com/thepirhana
# Last Update: 5-4-2020
# Project: Visualise t-SNE Iterations for NIFTY 50 Stocks
#          with User interactivity features
##########################################################

# ALL IMPORTS
import numpy as np
from numpy import linalg
import sklearn
from sklearn.manifold import TSNE
from time import time
import pandas as pd
import numpy as np
import fileinput
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import grid, column, gridplot, layout, row
from bokeh.models import CustomJS, Slider, ColumnDataSource, CategoricalColorMapper, LabelSet, Title, Legend
from bokeh.models.widgets import Button, TableColumn, Toggle
from bokeh.palettes import inferno, magma, viridis, gray, cividis, turbo, d3, RdYlGn10

# SAVING SOME CONFIGURATIONS WHICH WE'LL BE NEEDING
NIFTY_50_TSNE_SAVED_POSITIONS = 'tf__nifty_tsne_output_positions.csv'
NIFTY_50_DATA_INPUT_DATA = 'nifty_50_3_april_2020_snapshot.csv'
OUTPUT_HTML_FILE = 'tf__tsne_bokeh_nifty.html'
BOKEH_API_CDN = '<script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-api-1.4.0.js"></script>'
BOKEH_TOOLS = "hover,save,pan,reset,wheel_zoom,box_select,tap,undo,redo,zoom_in,zoom_out,crosshair"
output_file(OUTPUT_HTML_FILE)
TOOLTIPS=[('(X , Y)', '(@x,@y)'), ('Company', '@label'), ('Sector', '@sector')]
PLOTWIDTH=900
PLOTHEIGHT=900
TSNE_POSITIONS_BY_ITERATIONS = []

def prepare_nifty_data_for_tsne_feed():
    """Reads raw NIFTY 50 Stocks data and prepares DATA Feed for TSNE Model."""

    # Reading data and mkaing DF from CSV
    nifty_df = pd.read_csv(NIFTY_50_DATA_INPUT_DATA)

    ## CONVERTING ALL DATA INTO NUMPY ARRAYS 
    # Storing all sectors 
    nifty_tsne_feed_target_sector = nifty_df[nifty_df.columns[0]].to_numpy()

    # Storing all companies
    nifty_tsne_feed_target = nifty_df[nifty_df.columns[1]].to_numpy()

    # Storing main data feed which is to be fed into TSNE
    nifty_tsne_feed = nifty_df[nifty_df.columns[2:]].to_numpy()
    
    return nifty_df, nifty_tsne_feed_target_sector, nifty_tsne_feed_target, nifty_tsne_feed

def carry_bokeh_correction():
    """Adds BOKEH-API JS import to created HTML PLOT, which is required as BOKEH currently misses to do so."""

    with fileinput.FileInput(OUTPUT_HTML_FILE, inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace('1.4.0.min', '1.4.0'), end='')

    with open(OUTPUT_HTML_FILE) as f:
        code = f.readlines() 

    new_code = []
    for line in code:
        new_code.append(line)
        if 'bokeh-widgets-1.4.0' in line:
            new_code.append(BOKEH_API_CDN)

    with open(OUTPUT_HTML_FILE,'w') as f:
        f.writelines(new_code)

def _gradient_descent(objective, p0, it, n_iter,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7, verbose=0, args=None, kwargs=None):
    """Batch gradient descent with momentum and individual gains.
    """

    """Function from sklearn that needs to be Monkey Patched."""

    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    ################MONKEY PATCH START
    count=0
    ################MONKEY PATCH END
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = it

    tic = time()
    for i in range(it, n_iter):
        count+=1
        ################MONKEY PATCH START
        if count%10 == 0:
            TSNE_POSITIONS_BY_ITERATIONS.append(p.copy())
        ################MONKEY PATCH END
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs['compute_error'] = check_convergence or i == n_iter - 1

        error, grad = objective(p, *args, **kwargs)
        grad_norm = linalg.norm(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break

    return p, error, i

def prepare_nifty_50_positions_data_from_tsne(nifty_df, nifty_tsne_feed_target_sector, nifty_tsne_feed_target, nifty_tsne_feed):
    """Applies TSNE to the NIFTY50 Data, preprocesses the output and stores in CSV."""

    # BEGIN TSNE
    X_proj_nifty = TSNE(n_iter=500).fit_transform(nifty_tsne_feed)

    # RESHAPING TSNE-POSITIONAL DATA TO GET INTO 2-D FORMAT
    X_iter_nifty = np.dstack(position.reshape(-1, 2) for position in TSNE_POSITIONS_BY_ITERATIONS)

    # PREPARING INDEX TO BE GIVEN FOR EACH ITERATION
    index = np.concatenate([np.repeat([i], 51) for i in range(50)])

    # INITIALIZING DF WITH INDEX
    df = pd.DataFrame(index=index)

    # ADDING ALL ESSENTIAL COLUMNS TO DF one-by-one
    df['X-coordinate'] = X_iter_nifty[:, 0, :].T.reshape((-1))
    df['Y-coordinate'] = X_iter_nifty[:, 1, :].T.reshape((-1))
    df['Label'] = np.concatenate([nifty_tsne_feed_target for i in range(50)])
    df['Sector'] = np.concatenate([nifty_tsne_feed_target_sector for i in range(50)])

    # SAVING DF
    df.to_csv(NIFTY_50_TSNE_SAVED_POSITIONS)

def processing_nifty_50_positions_data_for_plot():
    """Preparing DATA Source for feeding into BOKEH PLOT."""

    df_nifty_tsne = pd.read_csv(NIFTY_50_TSNE_SAVED_POSITIONS,index_col=[0])
    df_nifty_tsne.rename(columns={"X-coordinate": 'x', "Y-coordinate": 'y', "Label": 'label', "Sector": 'sector'}, inplace = True)
    
    # WE NEED THIS COLUMN TO SELECT THE ITERATION THAT WE WANT
    # TO VIEW IN THE PLOT
    df_nifty_tsne['Slice'] = df_nifty_tsne.index

    # WE NEED TO CONVERT LABEL COLUMN TO STRING TYPE
    # AS BOKEH NEEDS CATEGORICAL DATA TO BE IN STRING FORMAT
    df_nifty_tsne['label'] = df_nifty_tsne['label'].astype(str)

    # PREPARING THE FIRST VIEW DATA FOR PLOT TO BE 
    # RENDERED WHEN PLOT LOADS UP
    slice_data = df_nifty_tsne[df_nifty_tsne.Slice==0]
    dict_slice = {}
    dict_full = {}
    for col in df_nifty_tsne.columns:
        dict_slice[col] = slice_data[col]
        dict_full[col] = df_nifty_tsne[col]

    # PREPARING COLUMNDATASOURCE FOR BOKEH PLOT FEED
    sliceCDS = ColumnDataSource(dict_slice)
    fullCDS = ColumnDataSource(dict_full)

    # WE NEED THIS CDS FOR THE SLIDER IN PLOT
    indexCDS = ColumnDataSource(dict(index=[*range(0,50)]))

    return df_nifty_tsne, sliceCDS, fullCDS, indexCDS

def initializing_tsne_plot(df_nifty_tsne, sliceCDS, fullCDS, indexCDS):
    """Creating BOKEH Plot and giving it specifications."""

c

    return p

def consolidate_plot_and_save(df_nifty_tsne, sliceCDS, fullCDS, indexCDS):
    """Stitching the entire plot and adding interactivity."""

    ## CREATING A SLIDER FOR PLOT AND VIEWING TSNE ITERATIONS
    # SLIDER CALLBACK CUSTOM JS
    SliderCallback = CustomJS(args = dict(sliceCDS=sliceCDS, fullCDS=fullCDS, indexCDS=indexCDS), code = """
        const new_value = cb_obj.value;
        
        // Take the 'Slice' column from the full data
        const slice_col = fullCDS.data['Slice'];

        // Select only the values equal to the new slice number
        const mask = slice_col.map((item) => item==new_value);
        
        sliceCDS.data['y'] = fullCDS.data['y'].filter((item,idx) => mask[idx]);
        sliceCDS.data['x'] = fullCDS.data['x'].filter((item,idx) => mask[idx]);
        sliceCDS.data['label'] = fullCDS.data['label'].filter((item,idx) => mask[idx]);

        // Update the sliceCDS
        sliceCDS.change.emit();
        """)
    slider = Slider(title="Status at iteration (x10): ",start=0, end=49,value=0,step=1, height=40, width=int(PLOTWIDTH*0.9))
    slider.js_on_change('value', SliderCallback)

    ## CREATING A PLAY/PAUSE TOGGLE FOR PLOT FOR GOING THROUGH ALL TSNE ITERATIONS
    # PLAY/PAUSE CALLBACK CUSTOM JS
    PlayPauseToggleCallback = CustomJS(args=dict(slider=slider,indexCDS=indexCDS),code="""
    // A little lengthy but it works for me, for this problem, in this version.
        var check_and_iterate = function(index){
            var slider_val = slider.value;
            var toggle_val = cb_obj.active;
            if(toggle_val == false) {
                cb_obj.label = '► Play';
                clearInterval(looop);
                } 
            else if(slider_val == index[index.length - 1]) {
                cb_obj.label = '► Play';
                slider.value = index[0];
                cb_obj.active = false;
                clearInterval(looop);
                }
            else if(slider_val !== index[index.length - 1]){
                slider.value = index.filter((item) => item > slider_val)[0];
                }
            else {
            clearInterval(looop);
                }
        }
        if(cb_obj.active == false){
            cb_obj.label = '► Play';
            clearInterval(looop);
        }
        else {
            cb_obj.label = '❚❚ Pause';
            var looop = setInterval(check_and_iterate, 800, indexCDS.data['index']);
        };
    """)
    toggler = Toggle(label='► Play',active=False, height=40, width=int(PLOTWIDTH*0.1))
    toggler.js_on_change('active',PlayPauseToggleCallback)

    # ENCAPSULATING PLAY/PAUSE TOGGLER AND SLIDER
    widgets = row(toggler,slider, sizing_mode="fixed", height=40, width=PLOTWIDTH)

    # INTEGRATING WITH FINAL GRAPH
    final_graph = column(initializing_tsne_plot(df_nifty_tsne, sliceCDS, fullCDS, indexCDS),widgets)
    
    # SAVING THE FINAL GRAPH
    show(final_graph)

    # ADDING IMPORTS TO HTML PLOT
    carry_bokeh_correction()

if __name__ == "__main__": 

    # Monkey Patching the _gradient_descent function to store all positions
    sklearn.manifold._t_sne._gradient_descent = _gradient_descent

    # Preparing NIFTY 50 Data feed
    nifty_df, nifty_tsne_feed_target_sector, nifty_tsne_feed_target, nifty_tsne_feed = prepare_nifty_data_for_tsne_feed()

    # Carrying out TSNE and storing TSNE Positional data
    prepare_nifty_50_positions_data_from_tsne(nifty_df, nifty_tsne_feed_target_sector, nifty_tsne_feed_target, nifty_tsne_feed)

    # Preparing data feeds for BOKEH plot
    df_nifty_tsne, sliceCDS, fullCDS, indexCDS = processing_nifty_50_positions_data_for_plot()

    # Finally consolidating everything one roof and giving life to our plot
    consolidate_plot_and_save(df_nifty_tsne, sliceCDS, fullCDS, indexCDS)
