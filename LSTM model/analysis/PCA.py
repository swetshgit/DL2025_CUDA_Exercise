#Feature Selection using PCA
import sklearn
from sklearn.decomposition import PCA

#Method for applying PCA to find the features that retain the most variance

def apply_pca(df, n_components=0.95): #to retain 95% of the variance
    
    pca= PCA(n_components=n_components)
    principalComponents= pca.fit_transform(df)
    return pd.DataFrame(data=principalComponents), pca
