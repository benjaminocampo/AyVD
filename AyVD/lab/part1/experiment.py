import pandas as pd


URL = 'https://cs.famaf.unc.edu.ar/~mteruel/datasets/diplodatos/sysarmy_survey_2020_processed.csv'
DATASET = pd.read_csv(URL)


class Experiment:
    def __init__(self, relevant_cols):
        assert relevant_cols in DATASET.columns.values.tolist()
        self.relevant_cols = relevant_cols
        self.df = DATASET[relevant_cols]

    def filter_rows(self, cond):
        return self.df[cond]

    def add_cured_col(self, uncured_col, cured_col, cure_func):
        assert uncured_col in self.relevant_cols
        assert cured_col_name not in self.relevant_cols

        self.df.loc[:, cured_col] = self.df[uncured_col] \
            .apply(cure_func)

    def value_counts(self, col):
        return self.df[col].value_counts()

    # TODO: is there any other solution? TODO: document better TODO: if all the
    # instances of unstacked_col aren't lits it will keep the data frame
    # unchanched. But computes all the function. TODO: Change it so
    # unstacked_col only has instances of the list class.
    def stack_col(self, stacked_col, unstacked_col):
        self.df_lang = self.df[unstacked_col] \
            .apply(pd.Series).stack()\
            .reset_index(level=-1, drop=True).to_frame()\
            .join(self.df)\
            .rename(columns={0: stacked_col})
