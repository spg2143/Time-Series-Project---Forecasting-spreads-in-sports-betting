import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from matplotlib.dates import YearLocator, DateFormatter
import matplotlib.ticker as mt
import datetime as dt
import os

class setup():
    def __init__(self):
        self.setup = ""

    def filter_df(data_frame, column, condition, type='eq'):
        """
        Type can be: eq, gr, greq, le, leq, ue which stands for
        equal, greater, greater equal, less, less equal, unequal
        """
        if type == 'eq': 
            filter_df = data_frame[data_frame[column] == condition]
        elif type == 'gr':
            filter_df = data_frame[data_frame[column] > condition]
        elif type == 'greq':
            filter_df = data_frame[data_frame[column] >= condition]
        elif type == 'le':
            filter_df = data_frame[data_frame[column] < condition]
        elif type == 'leq':
            filter_df = data_frame[data_frame[column] <= condition]
        elif type == 'ue': 
            filter_df = data_frame[data_frame[column] != condition]
        elif type == 'in':
            filter_df = data_frame[data_frame[column].isin(condition)]
        return filter_df



    def custom_plot(
        x, y, data, type,
        title = '', x_label = '', y_label = '',
        color = None, palette = None, hue = None,
        remove_axes = ['top', 'right'], remove_legend = False,
        y_ticks = None, x_ticks = None, remove_ticks = False,
        date_plot = False, date_column = 'date',
        height = None, width = None, size_unit = 'cm',
        ylim = None, x_rotation = None, order = None, legend=None
    ):
        
        # import some libraries
        import warnings
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        import seaborn as sns
        import pandas as pd
        import numpy as np
        
        # set size
        cm = 1/2.54
        if (width != None) & (height != None):
            if size_unit == 'cm':
                plt.figure(figsize = (width*cm, height*cm))
            elif size_unit == 'inches':
                plt.figure(figsize = (width*cm, height*cm))
            else:
                warnings.warn('Unknown size unit. Please select one of the following: cm, inches')
        
        # make the plot        
        if type == 'bar':
            sns.barplot(
                x = x, y = y, data = data,
                color = color, palette = palette,
                ci = None, order = order
            )
        elif type == 'hue_bar':
            sns.barplot(
                x = x, y = y, data = data,
                color = color, palette = palette,
                ci = None, order = order, hue = hue
            )
        elif type == 'line':
            sns.lineplot(
                x = x, y = y, data = data,
                color = color, palette = palette
            )
        elif type == 'hue_line':
            sns.lineplot(
                x = x, y = y, data = data, hue = hue,
                color = color, palette = palette
            )
        elif type == 'hist': 
            sns.histplot(x=x, y=y, hue=hue, color=color
            )
        else:
            raise ValueError('Type of Plot must be "bar" or "line"')
        
        # set fonts and labels
        font = {'fontname': 'Arial Unicode MS'}
        plt.title(title, **font, color = '#4A4A4A')
        plt.xlabel(x_label, **font, color = '#4A4A4A')
        plt.ylabel(y_label, **font, color = '#4A4A4A')
        plt.xticks(**font, color = '#4A4A4A')
        plt.yticks(**font, color = '#4A4A4A')
        
        # get axes
        ax = plt.gca()
        
        # remove axes
        if remove_axes != False:
            if all(position in ['left', 'top', 'right', 'bottom'] for position in remove_axes):
                for i in remove_axes:
                    ax.spines[i].set_visible(False)
            else:
                warnings.warn('remove_axes parameter accepts either False or a list of the following options: ["left", "top", "right", "bottom"]')
        
        # remove legend
        if remove_legend:
            ax.get_legend().remove()
        
        # make ticks and labels on axis
        if y_ticks != None:
            ax.set_yticks(y_ticks)
        if (x_ticks != None) & (date_plot == False):
            ax.set_xticks(x_ticks)
        if date_plot & (type == 'bar'):
            x_ticks = np.arange(start = 3, stop = len(data)-1, step = len(data)/5, dtype = int)
            ax.set_xticks(x_ticks)
            x_labels = data.iloc[x_ticks][date_column].dt.strftime('%m-%Y')
            ax.set_xticklabels(labels = x_labels, rotation = 0);
        if remove_ticks & (type in ['bar', 'hue_bar']):
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticks_position('none')
        elif date_plot & (type in ['line', 'hue_line']):
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            help_data = data[['date']].drop_duplicates().sort_values('date')
            x_ticks_loc = np.arange(start = 3, stop = len(help_data)-1, step = len(help_data)/5, dtype = int)
            x_ticks = help_data.iloc[x_ticks_loc]['date']
            ax.set_xticks(x_ticks)
            date_form = DateFormatter('%Y-%m')
            ax.xaxis.set_major_formatter(date_form)
            
        if ylim is not None:
            ax.set(ylim = (ylim[0], ylim[1]))
            
        if x_rotation is not None:
            plt.xticks(rotation = x_rotation)


    def custom_scatter_plot(
        x, y, data,
        size = None, hue = None, palette = None, legend = True,
        title = '', x_label = '', y_label = '',
        remove_axes = ['top', 'right'], remove_ticks = True,
        highlight_variable = None, highlight_condition = None, bool_arg = 'eq', highlight_rgb = [175, 9, 37],
        non_highlight_rgb = [190, 190, 190], transparency = [1, 0.5],
        height = None, width = None, size_unit = 'cm',
        xlim = None, ylim = None, yticks = None, xticks = None
    ):
        
        # load libraries
        import warnings
        
        # set size
        cm = 1/2.54
        if (width != None) & (height != None):
            if size_unit == 'cm':
                plt.figure(figsize = (width*cm, height*cm))
            elif size_unit == 'inches':
                plt.figure(figsize = (width*cm, height*cm))
            else:
                warnings.warn('Unknown size unit. Please select one of the following: cm, inches')
        
        # make color codes for highlighted points
        if (highlight_variable != None) & (highlight_condition != None):
            red = np.where((data[highlight_variable] == highlight_condition), (highlight_rgb[0]/255), (non_highlight_rgb[0]/255))
            green = np.where((data[highlight_variable] == highlight_condition), (highlight_rgb[1]/255), (non_highlight_rgb[1]/255))
            blue = np.where((data[highlight_variable] == highlight_condition), (highlight_rgb[2]/255), (non_highlight_rgb[2]/255))
            transparency_scores = np.where((data[highlight_variable] == highlight_condition), transparency[0], transparency[1])
            # mapping for plot
            rgba_colors = np.zeros((len(transparency_scores), 4))
            rgba_colors[:, 0] = red
            rgba_colors[:, 1] = green
            rgba_colors[:, 2] = blue
            rgba_colors[:, 3] = transparency_scores
            
            # make the plot
            sns.scatterplot(
                x = x, y = y, data = data,
                size = size, sizes = (10, 200),
                c = rgba_colors,
                s = 20, linewidth = 0.2
            )
        
        elif highlight_variable == 'pattern_special_case':
            # special case: make the points outside the window in the demand patterns lighter
            transparency_scores = np.where(((data['cv2'] > 2.5) | (data['p'] > 2.5)), 0.15, 1)
            rgba_colors = np.zeros((len(transparency_scores), 4))
            rgba_colors[:, 0] = (175/255)
            rgba_colors[:, 1] = (9/255)
            rgba_colors[:, 2] = (37/255)
            rgba_colors[:, 3] = transparency_scores
            
            # make the plot
            sns.scatterplot(
                x = x, y = y, data = data,
                size = size, sizes = (10, 200),
                c = rgba_colors,
                s = 20, linewidth = 0.2
            )

        else:
            # make the plot
            sns.scatterplot(
                x = x, y = y, data = data,
                hue = hue, palette = palette,
                size = size, sizes = (10, 200),
                s = 20, linewidth = 0.2,
                color = '#C00A28', legend = legend
            )
        
        # set x and y
        if xlim != None:
            plt.xlim(xlim[0], xlim[1])
        if ylim != None:
            plt.ylim(ylim[0], ylim[1])
            
        # set titel and labels
        font = {'fontname': 'Arial Unicode MS'}
        plt.title(title, **font, color = '#4A4A4A')
        plt.xlabel(x_label, **font, color = '#4A4A4A')
        plt.ylabel(y_label, **font, color = '#4A4A4A')
        plt.xticks(**font, color = '#4A4A4A')
        plt.yticks(**font, color = '#4A4A4A')
        
        # get axes
        ax = plt.gca()
        
        # remove axes
        if remove_axes != False:
            if all(position in ['left', 'top', 'right', 'bottom'] for position in remove_axes):
                for i in remove_axes:
                    ax.spines[i].set_visible(False)
            else:
                warnings.warn('remove_axes parameter accepts either False or a list of the following options: ["left", "top", "right", "bottom"]')
        
        # remove legend
        if size != None:
            ax.get_legend().remove()
            
        # set x and y ticks
        if remove_ticks:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
        if yticks != None:
            ax.set_yticks(yticks);
        if xticks != None:
            ax.set_xticks(xticks);


    def create_plot_data(actuals_data, forecast_data, series_guid, slice_id, var_oi,
                        cutoff_date = None, scale = 1000):
        
        # modify actuals
        actuals_data = actuals_data \
            [['date', 'series_guid', var_oi]] \
            .rename(columns = {
                var_oi: 'value'  # if something doesnt work this is new and not tested!
            })
        actuals_data = filter_df(actuals_data, 'series_guid', series_guid, 'eq')
        actuals_data['type'] = 'actual'
        actuals_data['date'] = pd.to_datetime(actuals_data['date'])

        # modify forecasts
        forecast_data = filter_df(forecast_data, 'series_guid', series_guid, 'eq')
        forecast_data = filter_df(forecast_data, 'slice_id', slice_id, 'eq')
        forecast_data = forecast_data \
            [['date', 'prediction', 'series_guid']] \
                .rename(columns = {
                    'prediction': 'value'
                })
        forecast_data['type'] = 'prediction'
        
        final_df = pd.concat([actuals_data, forecast_data])
        
        final_df['date'] = pd.to_datetime(final_df['date'])
        
        if cutoff_date is not None:
            final_df = filter_df(final_df, 'date', cutoff_date, 'ge')
            
        if scale != 1:
            final_df['value_scaled'] = final_df['value']/scale
            
        final_df.reset_index(drop = True, inplace = True)
        
        return final_df


    def mkdir_p(mypath):
        from errno import EEXIST
        from os import makedirs,path

        try:
            makedirs(mypath)
        except OSError as exc: # Python >2.5
            if exc.errno == EEXIST and path.isdir(mypath):
                pass
            else: raise
            
    def custom_boxplot(
        y, data, color = '#DD0B2F',
        title = '', x_label = '', y_label = '',
        swarmplot = True, point_color = '.25', point_size = 4,
        remove_axes = ['top', 'right', 'left', 'bottom'], remove_ticks = True,
        height = None, width = None, size_unit = 'cm'
    ):
        # load libraries
        import warnings
        
        # set size
        cm = 1/2.54
        if (width != None) & (height != None):
            if size_unit == 'cm':
                plt.figure(figsize = (width*cm, height*cm))
            elif size_unit == 'inches':
                plt.figure(figsize = (width*cm, height*cm))
            else:
                warnings.warn('Unknown size unit. Please select one of the following: cm, inches')
    
        # make the plot
        sns.boxplot(y = y, data = data, color = color)
        if swarmplot:
            sns.swarmplot(y = y, data = data, color = point_color, size = point_size)
            
        # get axes
        ax = plt.gca()
        
        # remove axes
        if remove_axes != False:
            if all(position in ['left', 'top', 'right', 'bottom'] for position in remove_axes):
                for i in remove_axes:
                    ax.spines[i].set_visible(False)
            else:
                warnings.warn('remove_axes parameter accepts either False or a list of the following options: ["left", "top", "right", "bottom"]')
                
        # remove grid lines
        ax.grid(False)
        
        # set labels
        font = {'fontname': 'Arial Unicode MS'}
        plt.title(title, **font, color = '#4A4A4A')
        plt.xlabel(x_label, **font, color = '#4A4A4A')
        plt.ylabel(y_label, **font, color = '#4A4A4A')
        plt.xticks(**font, color = '#4A4A4A')
        plt.yticks(**font, color = '#4A4A4A')
        
        # remove ticks
        if remove_ticks:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')


    def implicit_filling(data: pd.DataFrame, 
                        account_number: str):
        
        # Collect existing months for respective data
        existing_months = data.month.unique()
        months = range(1,13)

        # Create remaining months
        remaining_months = list(set(months) - set(existing_months))
        current_year = data.year.unique()

        # Create temp DataFrame with missing values
        temp_data = pd.DataFrame({'year': np.repeat(current_year, len(remaining_months)),
                                    'month': remaining_months, 'account_number':np.repeat(account_number, len(remaining_months)),
                                    'amount_quoted': np.repeat(0, len(remaining_months))})
        
        # Add and sort by month, return sorted Frame
        data_new = pd.concat([data, temp_data], axis=0)
        data_new = data_new.sort_values(by='month')
        return data_new


    def custom_histogram(data, x, bins = None, color='#DD0B2F', x_label = '', y_label = '', hue=None, palette=None,
                        xlim=None, ylim=None, y_ticks = None, x_ticks = None, remove_ticks = False,remove_legend = False,
                        remove_axes = ['top' ,'right'], title='', stat='count', discrete=True):

        import warnings

        sns.histplot(data=data, x=x, hue=hue, palette=palette, color=color, stat=stat,discrete=discrete, bins=bins)

            # set x and y
        if xlim != None:
            plt.xlim(xlim[0], xlim[1])
        if ylim != None:
            plt.ylim(ylim[0], ylim[1])

        # set titel and labels
        font = {'fontname': 'Arial'}
        plt.title(title, **font, color = '#4A4A4A')
        plt.xlabel(x_label, **font, color = '#4A4A4A')
        plt.ylabel(y_label, **font, color = '#4A4A4A')
        plt.xticks(**font, color = '#4A4A4A')
        plt.yticks(**font, color = '#4A4A4A')
        
        # get axes
        ax = plt.gca()
        
        # remove axes
        if remove_axes != False:
            if all(position in ['left', 'top', 'right', 'bottom'] for position in remove_axes):
                for i in remove_axes:
                    ax.spines[i].set_visible(False)
            else:
                warnings.warn('remove_axes parameter accepts either False or a list of the following options: ["left", "top", "right", "bottom"]')
        
        # remove legend
        if remove_legend:
            ax.get_legend().remove()
        
        # make ticks and labels on axis
        if y_ticks != None:
            ax.set_yticks(y_ticks)
        if x_ticks != None:
            ax.set_xticks(x_ticks)
        if remove_ticks & (type in ['bar', 'hue_bar']):
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticks_position('none')

    def filter_date_ts(data, date):

        #data['date'] = pd.to_datetime(data['date'])
        filter = data['date'] >= date
        data = data[filter]

        return data