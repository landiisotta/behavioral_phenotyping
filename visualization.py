import umap
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, \
    ColorBar, HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.plotting import figure, show, output_notebook
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import pi


class Visualization:
    """Class for the visualization of data and results. It returns:
    scatter plot, dendrogram, heatmap.
    """

    def __init__(self, subject_info, col_dict, c_out):
        """
        Parameters
        ----------
        subject_info: dictionary
            Dictionary with subject demographics (Pinfo dataclass)
            as returned by cohort_info method in dataset module
        """
        self.c_out = c_out  # List of colors to exclude
        self.col_dict = col_dict  # Dictionary of colors from matplotlib
        colormap = [c for c in self.col_dict if c not in self.c_out]
        self.colormap = colormap
        self.subject_info = subject_info

    @staticmethod
    def data_scatter_dendrogram(X,
                                subc_dict,
                                pid_list=None,
                                **kwargs):
        """ Prepare the data to be visualized in umap scatterplot and
        dendrogram

        Parameters
        ----------
        X: array, dataframe
            either an array (as returned by patient embedding functions)
            or a dataframe (feature dataset)
        subc_dict: dictionary
            dictionary of pids and subcluster labels
        pid_list: list
            list of pids as ordered in X
        kwargs: kewyword arguments to be passed to UMAP

        Returns
        -------
        numpy array
            umap projection
        list
            list of tuple with pid and subcluster label
        """
        if isinstance(X, pd.DataFrame):
            pid_list = list(X.index)
            X = X.to_numpy()

        umap_mtx = umap.UMAP(**kwargs).fit_transform(X)

        return umap_mtx, [(pid, subc_dict[pid]) for pid in pid_list]

    def data_heatmap_feat(self, X, X_scaled, subc_dict):
        """ Prepare data as input to heatmap feature.

        Parameters
        ----------
        X: dataframe
            Dataframe with raw feature values
        X_scaled: dataframe
            Dataframe with scaled feature values
        subc_dict: dictionary
            Dictionary with subject ids and subcluster labels

        Returns
        -------
        dataframe
            Object with both scaled and row values. A column with subcluster and
            subject id is added.
        """
        X_scaled = pd.DataFrame(X_scaled.sort_index().stack(),
                                columns=['score_sc']).reset_index()
        X_scaled.columns = ['clpid', 'feat', 'score_sc']
        X_scaled['clpid'] = ['-'.join([str(subc_dict[pid]), pid])
                             for pid in X_scaled['clpid']]
        X_scaled = X_scaled.sort_values(by='feat')

        X = pd.DataFrame(X.sort_index().stack(), columns=['score']).reset_index()
        X.columns = ['pid', 'feat', 'score']
        X = X.sort_values(by='feat')

        X_scaled['score'] = X['score']

        X_scaled = self._modify_df(X_scaled)

        return X_scaled

    def data_heatmap_emb(self, X, vocab, subc_dict):
        """ Prepare data as input to heatmap embeddings.

        Parameters
        ----------
        X: dictionary
            BEHR dictionary
        vocab: dictionary
            bt_to_idx vocabulary
        subc_dict: dictionary
            Dictionary with pid and subcluster labels

        Returns
        -------
        dataframe
            Dataframe with raw scores and scaled scores for subclusters.
            clpid columns with joined subcluster label and pid.
        """
        # Build feature list
        c_lab = sorted(set(['::'.join(lab.split('::')[:-1])
                            for lab in vocab.keys()]))

        dict_age = {}
        for p, behr in X.items():
            for vect in behr:
                if (p, vect[1]) not in dict_age:
                    dict_age[(p, vect[1])] = {}
                for t in vect[2:]:
                    ss = t.split('::')
                    dict_age[(p, vect[1])].setdefault('::'.join(ss[:-1]),
                                                      list()).append(int(ss[-1]))
        # Create dataframe with cl-pid as index
        val_dict = {}
        indx = []
        for vect in sorted(list(dict_age.keys())):
            for f in c_lab:
                try:
                    if len(dict_age[vect][f]) == 1:
                        val_dict.setdefault(f, list()).extend(dict_age[vect][f])
                    else:  # Mean of scores if multiple score per assessment
                        val_dict.setdefault(f, list()).append(np.mean(dict_age[vect][f]))
                except KeyError:
                    val_dict.setdefault(f, list()).append(None)
            indx.append(('-'.join([str(subc_dict[vect[0]]), vect[0]]), vect[1]))

        # create dataframe with cl-pi as index
        emb_df = pd.DataFrame(val_dict, index=indx)
        emb_df_imp = emb_df.fillna(emb_df.mean(), inplace=False)

        scaler = StandardScaler()
        emb_df_scaled = scaler.fit_transform(emb_df_imp.values)
        emb_df_scaled = pd.DataFrame(emb_df_scaled, index=indx,
                                     columns=emb_df.columns)

        emb_df = pd.DataFrame(emb_df.stack(dropna=False),
                              columns=['score']).reset_index()
        emb_df_scaled = pd.DataFrame(emb_df_scaled.stack(),
                                     columns=['score_sc']).reset_index()
        emb_df_scaled['score'] = emb_df['score']
        emb_df_scaled.columns = ['cllab_aoa', 'feat', 'score_sc', 'score']

        emb_df_scaled = self._modify_df(emb_df_scaled)

        emb_df_scaled['clpid'] = [tup[0] for tup in emb_df_scaled['cllab_aoa']]
        emb_df_scaled['aoa'] = [tup[1] for tup in emb_df_scaled['cllab_aoa']]

        emb_df_scaled = emb_df_scaled.dropna()

        return emb_df_scaled

    def scatterplot_dendrogram(self,
                               X,
                               umap_mtx,
                               pid_subc_list,
                               fig_height,
                               fig_width):
        """Scatterplot and dendrogram for clustering.

        Parameters
        ----------
        X: numpy array or dendrogram
            dendrogram input
        umap_mtx: np array
            Umap projections of patients
        pid_subc_list: list of tuples
            list of pid and subclusters tuples as ordered in X
        fig_height, fig_width: int
        """

        subc_list = [el[1] for el in pid_subc_list]

        colors = [self.colormap[cl] for cl in sorted(list(set(subc_list)))]
        # Bokeh scatterplot
        self._scatter_plot(umap_mtx, pid_subc_list, colors, fig_width, fig_height)

        # Dendrogram
        linked = linkage(X, 'ward')
        # Color mapping
        dflt_col = "#808080"  # Unclustered gray
        # * rows in Z correspond to "inverted U" links that connect clusters
        # * rows are ordered by increasing distance
        # * if the colors of the connected clusters match, use that color for link
        link_cols = {}
        for idx, lidx in enumerate(linked[:, :2].astype(int)):
            c1, c2 = (link_cols[x] if x > len(linked) else colors[subc_list[x]]
                      for x in lidx)
            link_cols[idx + 1 + len(linked)] = c1 if c1 == c2 else dflt_col

        plt.figure(figsize=(fig_height, fig_width))
        dendrogram(Z=linked,
                   labels=np.array(subc_list),
                   color_threshold=None,
                   leaf_font_size=5, leaf_rotation=0,
                   link_color_func=lambda x: link_cols[x])
        plt.show()

    @staticmethod
    def heatmap_feat(X_scaled,
                     fig_height,
                     fig_width):
        """ Bokeh heatmap for the visualization of scaled scores in the
        different subclusters. Hovertool displaying subject info and raw
        scores.

        Parameters
        ----------
        X_scaled: dataframe
            Feature scaled scores
        fig_height, fig_width: int
        """

        colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2",
                  "#dfccce", "#ddb7b1", "#cc7878", "#933b41",
                  "#550b1d"]

        mapper = LinearColorMapper(palette=colors,
                                   low=X_scaled.score_sc.min(),
                                   high=X_scaled.score_sc.max())
        output_notebook()
        p = figure(x_range=sorted(list(set(X_scaled['clpid']))),
                   y_range=sorted(list(set(X_scaled['feat']))),
                   x_axis_location="above",
                   plot_width=fig_width,
                   plot_height=fig_height,
                   toolbar_location='below')

        TOOLTIPS = [('clpid', '@clpid'),
                    ('sex', '@sex'),
                    ('bdate', '@bdate'),
                    ('feat', '@feat'),
                    ('score', '@score'),
                    ('n_enc', '@n_enc')]

        p.add_tools(HoverTool(tooltips=TOOLTIPS))

        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.xaxis.major_label_text_font_size = "7pt"
        p.yaxis.major_label_text_font_size = "7pt"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = pi / 2

        p.rect(x="clpid", y="feat",
               width=1, height=1,
               source=X_scaled,
               fill_color={'field': 'score_sc',
                           'transform': mapper},
               line_color=None)

        color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                             ticker=BasicTicker(desired_num_ticks=len(colors)),
                             formatter=PrintfTickFormatter(format="%d.2"),
                             label_standoff=6, border_line_color=None, location=(0, 0))
        p.add_layout(color_bar, 'right')
        show(p)

    @staticmethod
    def heatmap_emb(emb_df_scaled,
                    fig_height,
                    fig_width):
        """ Bokeh heatmap of scaled scores for patient embedding subclusters.
        Hovertool with subject info and subject raw scores.

        Parameters
        ----------
        emb_df_scaled: dataframe
            output of data_heatmap_emb
        fig_height, fig_width: int
        """

        colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2",
                  "#dfccce", "#ddb7b1", "#cc7878", "#933b41",
                  "#550b1d"]

        mapper = LinearColorMapper(palette=colors,
                                   low=emb_df_scaled.score_sc.min(),
                                   high=emb_df_scaled.score_sc.max())

        # output_notebook()
        p = figure(x_range=sorted(list(set(emb_df_scaled['clpid']))),
                   y_range=sorted(list(set(emb_df_scaled['feat']))),
                   x_axis_location="above",
                   plot_width=fig_width,
                   plot_height=fig_height,
                   toolbar_location='below')

        TOOLTIPS = [('clpid', '@clpid'),
                    ('sex', '@sex'),
                    ('bdate', '@bdate'),
                    ('aoa', '@aoa'),
                    ('feat', '@feat'),
                    ('score', '@score'),
                    ('n_enc', '@n_enc')]

        p.add_tools(HoverTool(tooltips=TOOLTIPS))

        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.xaxis.major_label_text_font_size = "7pt"
        p.yaxis.major_label_text_font_size = "7pt"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = pi / 2

        p.rect(x="clpid", y="feat",
               width=1, height=1,
               source=emb_df_scaled,
               fill_color={'field': 'score_sc',
                           'transform': mapper},
               line_color=None)

        color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                             ticker=BasicTicker(desired_num_ticks=len(colors)),
                             formatter=PrintfTickFormatter(format="%d.2"),
                             label_standoff=6, border_line_color=None, location=(0, 0))
        p.add_layout(color_bar, 'right')
        show(p)

    def _scatter_plot(self,
                      umap_mtx,
                      pid_subc_list,
                      colors,
                      fig_height,
                      fig_width):
        """Bokeh scatterplot to visualize in jupyter clusters and subject info.

        Parameters
        ----------
        umap_mtx: np array
            Array with UMAP projections
        pid_subc_list: list of tuples
            list of pids ordered as in umap_mtx and subcluster labels
        colors: list
            Color list
        fig_height, fig_width: int
            Figure dimensions
        """

        pid_list = list(map(lambda x: x[0], pid_subc_list))
        subc_list = list(map(lambda x: x[1], pid_subc_list))

        source = ColumnDataSource(dict(
            x=umap_mtx[:, 0].tolist(),
            y=umap_mtx[:, 1].tolist(),
            pid=pid_list,
            subc=list(map(lambda x: str(x), subc_list)),
            bdate=[self.subject_info[pid].dob for pid in pid_list],
            sex=[self.subject_info[pid].sex for pid in pid_list],
            n_enc=[self.subject_info[pid].n_enc for pid in pid_list]))

        cmap = CategoricalColorMapper(factors=[str(lab) for lab in range(len(set(subc_list)))],
                                      palette=colors)
        TOOLTIPS = [('pid', '@pid'),
                    ('subc', '@subc'),
                    ('sex', '@sex'),
                    ('bdate', '@bdate'),
                    ('n_enc', '@n_enc')]

        plotTools = 'box_zoom, wheel_zoom, pan,  crosshair, reset, save'

        output_notebook()
        p = figure(plot_width=fig_width * 50, plot_height=fig_height * 50,
                   tools=plotTools)
        p.add_tools(HoverTool(tooltips=TOOLTIPS))
        p.circle('x', 'y', legend='subc', source=source,
                 color={"field": 'subc', "transform": cmap})
        p.xaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None
        p.yaxis.major_tick_line_color = None
        p.yaxis.minor_tick_line_color = None
        p.xaxis.major_label_text_color = None
        p.yaxis.major_label_text_color = None
        p.grid.grid_line_color = None
        show(p)

    def _modify_df(self, df):
        """ Adds subject info to dataframe for heatmaps

        Parameters
        ----------
        df: dataframe
            Stacked scaled dataframe with cl-pid column

        Returns
        -------
        dataframe
            Dataframe with subject demographic info and number
            of encounters
        """

        sex_vect = []
        bdate_vect = []
        n_enc_vect = []
        for pid in df.iloc[:, 0]:
            if isinstance(pid, str):
                slab = pid.split('-')[1]
                sex_vect.append(self.subject_info[slab].sex)
                bdate_vect.append(self.subject_info[slab].dob)
                n_enc_vect.append(self.subject_info[slab].n_enc)
            else:
                slab = pid[0].split('-')[1]
                sex_vect.append(self.subject_info[slab].sex)
                bdate_vect.append(self.subject_info[slab].dob)
                n_enc_vect.append(self.subject_info[slab].n_enc)

        df['sex'] = sex_vect
        df['bdate'] = bdate_vect
        df['n_enc'] = n_enc_vect

        return df
