import umap
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, \
    ColorBar, HoverTool, ColumnDataSource, CategoricalColorMapper, CustomJS
from bokeh.plotting import figure, show, output_notebook
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import pi


class Visualization:
    
    def __init__(self, subject_info, col_dict, c_out):
        self.subject_info = subject_info
        self.col_dict = col_dict
        self.c_out = c_out

    def _scatter_plot(self, umap_mtx, subc_dict, colors,
                      fig_height, fig_width):
        """bokeh scatterplot

        :param: X list or dataframe with embeddings or features
        :return:
        """
        subc_lab = [str(lab) for lab in subc_dict.values()]

        source = ColumnDataSource(dict(
            x=umap_mtx[:, 0].tolist(),
            y=umap_mtx[:, 1].tolist(),
            pid=list(subc_dict.keys()),
            subc=subc_lab,
            bdate=[self.subject_info[pid].bdate for pid in subc_dict.keys()],
            sex=[self.subject_info[pid].sex for pid in subc_dict.keys()],
            n_enc=[self.subject_info[pid].n_enc for pid in subc_dict.keys()]))

        cmap = CategoricalColorMapper(factors=sorted(list(set(subc_lab))),
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

        sex_vect = []
        bdate_vect = []
        n_enc_vect = []
        for pid in df.iloc[:, 0]:
            if isinstance(pid, str):
                slab = pid.split('-')[1]
                sex_vect.append(self.subject_info[slab].sex)
                bdate_vect.append(self.subject_info[slab].bdate)
                n_enc_vect.append(self.subject_info[slab].n_enc)
            else:
                slab = pid[0].split('-')[1]
                sex_vect.append(self.subject_info[slab].sex)
                bdate_vect.append(self.subject_info[slab].bdate)
                n_enc_vect.append(self.subject_info[slab].n_enc)

        df['sex'] = sex_vect
        df['bdate'] = bdate_vect
        df['n_enc'] = n_enc_vect

        return df

    def scatterplot_dendrogram(self, X, subc_dict,
                               fig_height, fig_width):

        # if X is dataframe, transform X in numpy array
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        colormap = [c for c in self.col_dict if c not in self.c_out]
        colormap_rid = [colormap[cl] for cl in sorted(list(set(subc_dict.values())))]
        # colors_en = [colormap_rid[v] for v in subc_dict.values()]
        umap_mtx = umap.UMAP(random_state=42).fit_transform(X)
        # plot bokeh scatterplot
        self._scatter_plot(umap_mtx, subc_dict, colormap_rid, fig_width, fig_height)

        linked = linkage(X, 'ward')
        # Color mapping
        dflt_col = "#808080"  # Unclustered gray
        # * rows in Z correspond to "inverted U" links that connect clusters
        # * rows are ordered by increasing distance
        # * if the colors of the connected clusters match, use that color for link
        link_cols = {}
        for idx, lidx in enumerate(linked[:, :2].astype(int)):
            c1, c2 = (link_cols[x] if x > len(linked) else colormap_rid[list(subc_dict.values())[x]]
                      for x in lidx)
            link_cols[idx + 1 + len(linked)] = c1 if c1 == c2 else dflt_col

        plt.figure(figsize=(fig_height, fig_width))
        # Dendrogram
        dendrogram(Z=linked, labels=list(subc_dict.values()),
                   color_threshold=None,
                   leaf_font_size=5, leaf_rotation=0,
                   link_color_func=lambda x: link_cols[x])
        plt.show()

    def heatmap_feat(self, X, X_scaled,
                     subc_dict,
                     fig_height, fig_width):
        """

        :param X: dataframe (index=pid)
        :param X_scaled: dataframe of scaled scores (index=pid)
        :param subc_dict: dictionary {pid: subc_lab}
        :param fig_height: int
        :param fig_width: int
        """

        colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2",
                  "#dfccce", "#ddb7b1", "#cc7878", "#933b41",
                  "#550b1d"]

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

    def heatmap_emb(self, X, vocab, subc_dict,
                    fig_height, fig_width):
        """

        :param: X dictionary {pid: [(aoa, [vocab_idx])]}
        :param: vocab dictionary (idx_to_mt)
        :param: subc_dict dictionary {pid: subc_lab}
        :param: fig_height int
        :param: fig_width int
        """

        colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2",
                  "#dfccce", "#ddb7b1", "#cc7878", "#933b41",
                  "#550b1d"]

        # build feature list
        c_lab = sorted(set(['::'.join(lab.split('::')[:-1])
                            for lab in vocab.values()]))

        dict_age = {}
        for p, vect in X.items():
            for tup in vect:
                if (p, tup[0]) not in dict_age:
                    dict_age[(p, tup[0])] = {}
                for t in tup[1]:
                    ss = vocab[t].split('::')
                    dict_age[(p, tup[0])].setdefault('::'.join(ss[:-1]),
                                                     list()).append(int(ss[-1]))

        val_dict = {}
        indx = []
        for tup in sorted(list(dict_age.keys())):
            for f in c_lab:
                try:
                    if len(dict_age[tup][f]) == 1:
                        val_dict.setdefault(f, list()).extend(dict_age[tup][f])
                    else:
                        val_dict.setdefault(f, list()).append(np.mean(dict_age[tup][f]))
                except KeyError:
                    val_dict.setdefault(f, list()).append(None)
            indx.append(('-'.join([str(subc_dict[tup[0]]), tup[0]]), tup[1]))

        # create dataframe with cl-pi as index
        emb_df = pd.DataFrame(val_dict, index=indx)
        emb_df_imp = emb_df.fillna(emb_df.mean(), inplace=False)

        scaler = StandardScaler()
        emb_df_scaled = scaler.fit_transform(emb_df_imp.values)
        emb_df_scaled = pd.DataFrame(emb_df_scaled, index=indx,
                                     columns=emb_df.columns)

        emb_df = pd.DataFrame(emb_df.stack(dropna=False), columns=['score']).reset_index()
        emb_df_scaled = pd.DataFrame(emb_df_scaled.stack(), columns=['score_sc']).reset_index()
        emb_df_scaled['score'] = emb_df['score']
        emb_df_scaled.columns = ['cllab_aoa', 'feat', 'score_sc', 'score']

        emb_df_scaled = self._modify_df(emb_df_scaled)

        emb_df_scaled['clpid'] = [tup[0] for tup in emb_df_scaled['cllab_aoa']]
        emb_df_scaled['aoa'] = [tup[1] for tup in emb_df_scaled['cllab_aoa']]

        emb_df_scaled = emb_df_scaled.dropna()

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
