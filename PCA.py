import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
df = pd.read_csv(r'Data/SC_expression.csv')
# Now treatments are rows and genes are columns

# Remove gene names row
#df = df.drop(['Unnamed: 0'])
#df = df.set_index('IFFABF').transpose()
df = df.set_index('Genes')
cols = list(df.columns)
treatments = []
for col in cols:
  treatments.append(col[:3])
df = df.T
# PCA magic
pca = PCA(n_components=2, svd_solver='full')
pca.fit(df)
T = pca.transform(df)
pc_df = pd.DataFrame(data=T, columns=['PC1', 'PC2'])
pc_df = pc_df.assign(treatments=treatments)
ax = sns.lmplot(x="PC1", y="PC2",
  data=pc_df,
  fit_reg=False,
  hue='treatments',
  legend=True,
  scatter_kws={"s": 80}) # specify the point size

print(pc_df)
plt.show()


