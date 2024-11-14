# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:23:25 2024

@author: lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams['font.family'] = 'Verdana'
def plot_data(ax, model_data, title):
    # Bar locations
    colors = {'All': '#0570b0',
              'Easy': '#f6eff7',
              'Medium': '#bdc9e1',
              'Hard': '#67a9cf'
              }
    #['#0570b0', '#f6eff7', '#bdc9e1', '#67a9cf']
    x = range(len(model_data['Method'].unique()))
    custom_bar_width = 0.2  # Custom bar width
    y = range(20, 100 ,10)
    # Create bars for each difficulty with custom colors
    difficulties = ['All', 'Easy', 'Medium', 'Hard']
    offsets = [-1.5*custom_bar_width, -0.5*custom_bar_width, 0.5*custom_bar_width, 1.5*custom_bar_width]

    for diff, offset in zip(difficulties, offsets):
        ax.bar(x=[pos + offset for pos in x], height=model_data[diff], width=custom_bar_width,
               color=colors[diff], label=diff, edgecolor='grey')

    # Set plot details
    ax.set_title(title, fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_data['Method'].unique(), fontsize=15)
    ax.set_yticks(range(0, 81, 10))  # Set y-ticks
    ax.set_yticklabels(range(0, 81, 10), fontsize=15)  # Set y-ticks font size
    ax.set_ylabel('Accuracy(%)', fontsize=15)
    ax.set_ylim([25, 80])  # Custom y-axis limits
    #ax.legend(title='Difficulty', fontsize=15)
    ax.grid(True, linestyle='--', alpha=0.5)

def plot_method1(data):
    # Separate data for GPT-4V and Gemini-Pro
    gpt4v_data = data[data['Model'] == 'GPT-4V']
    gemini_pro_data = data[data['Model'] == 'Gemini-Pro']
    
    # Set up the figure and axes
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 6), sharey=False)
    
    # Function to plot data
    # Plot data for each model
    plot_data(axes[0], gpt4v_data, 'GPT-4V')
    
    plot_data(axes[1], gemini_pro_data, 'Gemini-Pro')
    colors = ['#0570b0', '#f6eff7', '#bdc9e1', '#67a9cf']
    ranges = ['All', 'Easy', 'Medium', 'Hard']
    legend_elements = [Patch(facecolor=color, label=range_item, edgecolor='black',)
                       for color, range_item in zip(colors, ranges)]
    
    fig.legend(handles=legend_elements,
               bbox_to_anchor=(0.5, 1.1),
               loc='upper center',
               fontsize=15, ncol=4)
    
    plt.tight_layout()
    plt.show()
    
def plot_data_combined_models(ax, gpt4v_data, gemini_pro_data, difficulty):
    # Bar locations
    x = range(len(gpt4v_data['Method'].unique()))
    bar_width = 0.4  # Custom bar width

    # Calculate offsets for side-by-side bars
    offsets = [-bar_width/2, bar_width/2]

    # Create bars for GPT-4V
    ax.bar([pos + offsets[0] for pos in x], height=gpt4v_data[difficulty], width=bar_width,
           color='#f6eff7', label='GPT-4V', edgecolor='grey')

    # Create bars for Gemini-Pro
    ax.bar([pos + offsets[1] for pos in x], height=gemini_pro_data[difficulty], width=bar_width,
           color='#bdc9e1', label='Gemini-Pro', edgecolor='grey')

    # Set plot details
    ax.set_title(f'{difficulty}', fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(gpt4v_data['Method'].unique(), fontsize=24)
    #ax.set_xlabel('Method', fontsize=14)
    #ax.set_yticklabels(range(0, 81, 10), fontsize=14)  # Set y-ticks font size
    
    ax.grid(True, linestyle='--', alpha=0.5)  # Add grid lines
    if difficulty == 'Hard':
        ax.set_ylim([25, 55])
        ax.set_yticks(range(25, 56, 10))
    elif difficulty == 'All':
        ax.set_ylabel('Accuracy(%)', fontsize=24)
        ax.set_ylim([45, 65])
        ax.set_yticks(range(45, 66, 5))
    elif difficulty == 'Easy':
        ax.set_ylim([60, 80])
        ax.set_yticks(range(55, 81, 10))
    elif difficulty == 'Medium':
        ax.set_ylim([45, 60])
        ax.set_yticks(range(45, 61, 5))

    # Setting y-tick labels with custom font size
    y_ticks = ax.get_yticks()
    ax.set_yticklabels([f"{int(tick)}" for tick in y_ticks], fontsize=24)
    
    
def plot_method2(data):
    gpt4v_data = data[data['Model'] == 'GPT-4V']
    gemini_pro_data = data[data['Model'] == 'Gemini-Pro']
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 4))  # Adjusted for four subplots

    difficulties = ['All', 'Easy', 'Medium', 'Hard']

    for i, diff in enumerate(difficulties):
        #plot_data_combined_models(axes[i // 2, i % 2], gpt4v_data, gemini_pro_data, diff)
        plot_data_combined_models(axes[i], gpt4v_data, gemini_pro_data, diff)

    
    colors = ['#f6eff7', '#bdc9e1']
    ranges = ['GPT-4V', 'Gemini-Pro']
    legend_elements = [Patch(facecolor=color, label=range_item, edgecolor='black',)
                       for color, range_item in zip(colors, ranges)]
    
    fig.legend(handles=legend_elements,
               bbox_to_anchor=(0.5, 1.18),
               loc='upper center',
               fontsize=24, ncol=2)
    
    plt.tight_layout()
    plt.savefig('MMMU.pdf', dpi=800, bbox_inches='tight')
    
    plt.show()
    
    
if __name__ == '__main__':
    # Load the Excel file to check its structure
    file_path = 'mmmu_data.xlsx'
    data = pd.read_excel(file_path)
    
    
    # Fill missing values in 'Model' column to ensure data integrity
    data['Model'].fillna(method='ffill', inplace=True)
    #plot_method1(data)
    plot_method2(data)
    
    

