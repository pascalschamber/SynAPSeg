import os
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class PlotConfig:
    """
    Configuration schema defining the data, aesthetics, and statistical operations 
    for a single multi-axis figure.
    
    This class acts as a single source of truth for how a specific dataset should 
    be visualized, effectively replacing deeply nested and unstructured dictionaries.

    Attributes:
        data (pd.DataFrame): 
            The specific slice of data to be plotted in this figure. This should 
            be pre-filtered (e.g., via `df.query()`) for the specific conditions, 
            ROIs, or timepoints desired.
            
        y_props (List[str]): 
            A list of column names in `data` to be plotted on the Y-axes. The length 
            of this list determines the total number of subplots generated in the figure.
            Example: `['density', 'intensity_mean', 'count']`
            
        x_col (str): 
            The name of the column in `data` to use for the X-axis across all subplots.
            Example: `'roi_name'` or `'DPI'`
            
        hue_col (Optional[str]): 
            The column name used to group and color the data. Typically represents 
            the primary experimental variable. 
            Default: None. Example: `'treatment'` or `'genotype'`
            
        base_plot_func (Callable): 
            The primary seaborn or matplotlib plotting function used as the underlying 
            layer (e.g., bars, boxes, or lines). 
            Default: `sns.barplot`. Alternatives: `sns.boxplot`, `sns.lineplot`.
            
        base_plot_kwargs (Dict[str, Any]): 
            Additional keyword arguments passed directly to the `base_plot_func`. 
            Useful for tweaking the specific aesthetics of the base layer without 
            affecting the overlays.
            Default: `{}`. Example: `{'capsize': 0.1, 'errwidth': 1.5}`
            
        swarm_kwargs (Dict[str, Any]): 
            Additional keyword arguments passed to the overlay scatter function 
            (`sns.swarmplot` or `sns.stripplot`). 
            Default: `{}`. Example: `{'size': 3, 'alpha': 0.6}`
            
        x_order (Optional[List[str]]): 
            An explicit list defining the left-to-right order of categories on the X-axis. 
            If None, defaults to the unique values found in `data[x_col]`.
            Default: None. Example: `['CA1', 'CA3', 'DG']`
            
        hue_order (Optional[List[str]]): 
            An explicit list defining the left-to-right order (and legend order) of 
            the hue categories. 
            Default: None. Example: `['Vehicle', 'Low Dose', 'High Dose']`
            
        palette (Optional[Dict[str, Any]]): 
            A dictionary mapping the unique values in `hue_col` to specific colors 
            (hex codes or standard color names). Ensures color consistency across figures.
            Default: None. Example: `{'Control': '#cccccc', 'Drug': '#ff0000'}`
            
        x_label (Optional[str]): 
            A custom string to display as the X-axis label. If None, the X-axis label 
            will just be the `x_col` string (or omitted depending on the generator logic).
            Default: None.
            
        split_by_sex (bool): 
            If True, intercepts the standard overlay point generation and splits the data 
            using the 'sex' column (assuming it exists), plotting Females and Males with 
            distinct markers (e.g., 'x' and 'o'). 
            Default: False.
            
        run_stats (bool): 
            If True, triggers the `stats_callback` function defined in the `FigureGenerator` 
            for each subplot. The resulting statistical summary is injected into the 
            subplot's title.
            Default: False.
            
        stats_kwargs (Dict[str, Any]): 
            A dictionary of keyword arguments passed down to the injected `stats_callback` 
            function. Typically includes the column names indicating the groups to compare.
            Default: `{}`. Example: `{'treatment_col': 'treatment', 'group1': 'Veh', 'group2': 'Drug'}`
    """
    data: pd.DataFrame
    y_props: List[str]
    x_col: str
    hue_col: Optional[str] = None
    
    # Plotting Functions
    base_plot_func: Callable = sns.barplot
    base_plot_kwargs: Dict[str, Any] = field(default_factory=dict)
    swarm_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Appearance and Data handling
    x_order: Optional[List[str]] = None
    hue_order: Optional[List[str]] = None
    palette: Optional[Dict[str, Any]] = None
    x_label: Optional[str] = None
    
    # Overlays & Stats
    split_by_sex: bool = False
    run_stats: bool = False
    stats_kwargs: Dict[str, Any] = field(default_factory=dict)


class FigureGenerator:
    """
    A wrapper around seaborn/matplotlib to generate multi-axis figures 
    based on a defined configuration schema.
    """
    
    def __init__(
        self, 
        save_dir: Optional[str] = None,
        save_ext: str = "svg",
        stats_callback: Optional[Callable] = None
    ):
        """
        Args:
            save_dir: Directory to save generated figures. If None, figures aren't saved.
            save_ext: File extension for saving figures (e.g., 'svg', 'png', 'pdf').
            stats_callback: A custom function to run stats. It should accept 
                            (df, prop, **kwargs) and return a formatted string/dict.
        """
        self.save_dir = save_dir
        self.save_ext = save_ext
        self.stats_callback = stats_callback

    def _get_safe_kwargs(self, func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitizes keyword arguments to prevent crashing incompatible seaborn functions 
        (e.g., passing 'dodge' to sns.lineplot).
        """
        safe_kwargs = kwargs.copy()
        sig_params = inspect.signature(func).parameters
        
        # Remove dodge if the target function doesn't explicitly accept it
        if 'dodge' not in sig_params and 'dodge' in safe_kwargs:
            safe_kwargs.pop('dodge')
            
        return safe_kwargs

    def _add_scatter_overlay(
        self, ax: plt.Axes, df: pd.DataFrame, config: PlotConfig, y_prop: str
    ) -> None:
        """Adds observation-level points (swarm or strip) over the base plot."""
        common_kwargs = {
            'data': df, 'x': config.x_col, 'y': y_prop, 'ax': ax,
            'hue': config.hue_col, 'dodge': True, 'palette': config.palette,
            'order': config.x_order, 'hue_order': config.hue_order
        }

        if config.split_by_sex and 'sex' in df.columns:
            # Custom split by sex logic
            for sex, marker in zip(['M', 'F'], ['o', 'x']):
                subdf = df[df['sex'] == sex]
                sns.stripplot(
                    **{**common_kwargs, **config.swarm_kwargs},
                    data=subdf, marker=marker, jitter=True,
                    zorder=9999, size=4, linewidth=1, edgecolor='k', alpha=0.4
                )
        else:
            # Standard swarm plot logic
            sns.swarmplot(
                **{**common_kwargs, **config.swarm_kwargs},
                zorder=9999, size=4, linewidth=1, edgecolor='k', alpha=0.8
            )

    def _handle_stats(
        self, ax: plt.Axes, df: pd.DataFrame, y_prop: str, config: PlotConfig
    ) -> Any:
        """Executes statistical testing and updates the axis title."""
        if config.run_stats and self.stats_callback:
            # Call custom injected stats function
            stats_result = self.stats_callback(df, y_prop, **config.stats_kwargs)
            
            # Assuming stats_result has a string representation or summary method
            title_text = str(stats_result).replace(": ", ":\n")
            ax.set_title(title_text)
            return stats_result
        return None

    def plot_single_figure(self, fig_name: str, config: PlotConfig, n_rows: int = 3) -> plt.Figure:
        """Generates a complete figure with multiple axes based on the config."""
        n_axes = len(config.y_props)
        n_cols = int(np.ceil(n_axes / n_rows))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        fig.suptitle(fig_name, y=1.02)
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        stats_results = {}

        # Default order fallback
        if config.x_order is None:
            config.x_order = config.data[config.x_col].unique().tolist()

        for i, y_prop in enumerate(config.y_props):
            ax = axes_flat[i]
            
            common_kwargs = {
                'data': config.data,
                'x': config.x_col,
                'y': y_prop,
                'ax': ax,
                'hue': config.hue_col,
                'order': config.x_order,
                'hue_order': config.hue_order,
                'palette': config.palette,
                'dodge': True,
                'legend': False,
            }

            # 1. Plot Base Layer
            base_kwargs = self._get_safe_kwargs(
                config.base_plot_func, 
                {**common_kwargs, **config.base_plot_kwargs}
            )
            config.base_plot_func(**base_kwargs)

            # 2. Plot Overlay Layer (Points)
            self._add_scatter_overlay(ax, config.data, config, y_prop)

            # 3. Handle Stats & Titles
            stats_out = self._handle_stats(ax, config.data, y_prop, config)
            if stats_out:
                stats_results[f"{fig_name}_{y_prop}"] = stats_out

            # 4. Format Labels
            ax.set_ylabel(y_prop.replace('_', ' ').title())
            if config.x_order:
                ax.set_xticks(np.arange(len(config.x_order)))
                ax.set_xticklabels(config.x_order, rotation=30, ha='center')
            if config.x_label:
                ax.set_xlabel(config.x_label)

        # Hide any unused subplots in the grid
        for j in range(n_axes, len(axes_flat)):
            axes_flat[j].set_visible(False)

        sns.despine(fig)
        plt.tight_layout()

        # Handle saving
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            safe_name = fig_name.replace(' ', '_').replace('=', '')
            filepath = os.path.join(self.save_dir, f"{safe_name}.{self.save_ext}")
            fig.savefig(filepath, bbox_inches='tight')
            print(f"Saved: {filepath}")

        return fig

    def plot_all(self, figures: Dict[str, PlotConfig], n_rows: int = 3) -> None:
        """Iterates through a dictionary of configurations and plots them all."""
        for name, config in figures.items():
            self.plot_single_figure(name, config, n_rows=n_rows)
            plt.show()

print('reloaded')

# =====================================================================
# Example Usage Implementation
# =====================================================================
if __name__ == "__main__":
    # 1. Define dummy data (in reality, this is your `andf` DataFrame)
    np.random.seed(42)
    andf = pd.DataFrame({
        'roi_name': np.random.choice(['CA1', 'CA3', 'DG'], 100),
        'treatment': np.random.choice(['Control', 'Drug'], 100),
        'sex': np.random.choice(['M', 'F'], 100),
        'colocal_id': np.random.choice([2, 3, 5, 6], 100),
        'DPI': np.random.choice([7, 11, 14], 100),
        'density': np.random.randn(100),
        'intensity_mean': np.random.randn(100) + 5,
        'region_area_mm': np.random.rand(100) * 10
    })

    # 2. Define shared properties
    _plt_props = ['density', 'intensity_mean', 'region_area_mm']
    PALETTE = {'Control': '#1f77b4', 'Drug': '#ff7f0e'}
    
    # Placeholder for your custom stat function
    def my_custom_stat_runner(df, prop, **kwargs):
        # e.g., utils_stats.check_ttest_assumptions_and_run(df, prop, **kwargs)
        return "p < 0.05 (Sig)"

    # 3. Create the configurations using the new Schema
    figures_schema = {
        'cFos properties by-animal': PlotConfig(
            data=andf.query("colocal_id in [3, 5, 6] & DPI in [14]"),
            y_props=_plt_props,
            x_col='roi_name',
            hue_col='treatment',
            base_plot_func=sns.barplot,
            palette=PALETTE,
            split_by_sex=True,
            run_stats=True,
            stats_kwargs={'treatment_col': 'treatment'}
        ),
        'cFos properties by DPI': PlotConfig(
            data=andf.query("colocal_id in [3, 5, 6]"),
            y_props=_plt_props,
            x_col='DPI',             # Switching the X axis dynamically
            hue_col='treatment',
            base_plot_func=sns.lineplot, # Safe! dodge will be automatically dropped
            palette=PALETTE,
            split_by_sex=False,
            run_stats=False
        )
    }

    # 4. Generate the figures
    generator = FigureGenerator(
        save_dir="./figures_out", 
        save_ext="svg",
        stats_callback=my_custom_stat_runner
    )
    generator.plot_all(figures_schema, n_rows=2)