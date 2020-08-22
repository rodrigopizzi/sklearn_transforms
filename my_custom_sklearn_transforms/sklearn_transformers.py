from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample
from sklearn.utils import shuffle


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class TransformacaoCustomizada:

    def fit(self):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Removendo dados vazios
        data = data.dropna()
        
        # Rebalanceando
        array_dificuldade = data.loc[data.PERFIL == "DIFICULDADE"]
        array_exatas = data.loc[data.PERFIL == "EXATAS"]
        array_excelente = data.loc[data.PERFIL == "EXCELENTE"]
        array_humanas = data.loc[data.PERFIL == "HUMANAS"]
        array_muito_bom = data.loc[data.PERFIL == "MUITO_BOM"]

        array_dificuldade2 = resample(array_dificuldade, n_samples=444)
        array_exatas2 = resample(array_exatas, n_samples=444)
        array_excelente2 = resample(array_excelente, n_samples=444)
        array_humanas2 = resample(array_humanas, n_samples=444)
        array_muito_bom2 = resample(array_muito_bom, n_samples=444)

        frames = [array_dificuldade2, array_exatas2, array_excelente2, array_humanas2, array_muito_bom2]
        result = pd.concat(frames)
        
        data = shuffle(result, random_state=123)
        
        return data