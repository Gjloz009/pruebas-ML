'''
Spaceship Titanic

-Codigo para acercarnos hacia el uso de 
las diferentes librerías sobre ML, en especifico
sckit-learn

La forma en que se usa como poner los valores tipo category 
si es con onehot encoder o con get dummies si cambai mucho


Los resultados máximos que conseguimos en accuracy es de:
0.7 aprox bajo un random forest sin modificar hiperparametros
en validación simple sin utilizar CV
-faltaría escoger hiperparametros con cv
-checar cuales son los atributos importantes que involucran al modelo
y seleccionar los mejores 
-falta hacer un solo pipeline que ejecute todo
'''
'''LIBRERIAS'''
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
'''END LIBRERIAS'''

'''FUNCIONES'''
def tweak_df(df):
    return(
        df.rename(columns=lambda c: c.lower())
                    .assign(
                    cabin_1 = lambda df_:df_.cabin.str.split('/',expand = True).iloc[:,2].fillna('S').astype('category'),
                    cabin_2 = lambda df_:df_.cabin.str.split('/',expand = True).iloc[:,0].fillna('B').astype('category'),
                    age = lambda df_:df_.age.fillna(df_.age.mean()),
                    roomservice = lambda df_:df_.roomservice.fillna(df_.roomservice.median()),
                    foodcourt = lambda df_:df_.foodcourt.fillna(df_.foodcourt.median()),
                    shoppingmall = lambda df_:df_.shoppingmall.fillna(df_.shoppingmall.median()),
                    spa = lambda df_:df_.spa.fillna(df_.spa.median()),
                    vrdeck = lambda df_:df_.vrdeck.fillna(df_.vrdeck.median()),
                    name = lambda df_:df_.name.fillna('Sin Nombre').astype('string'),
                    homeplanet = lambda df_:df_.homeplanet.fillna('Earth').astype('category'),
                    destination = lambda df_:df_.destination.fillna('TRAPPIST-1e').astype('category'),
                    cryosleep = lambda df_:df_.cryosleep.fillna('False').astype('bool'),
                    vip = lambda df_:df_.vip.fillna('False').astype('bool'),
                    total_expenses =  
                    lambda df_:df_.select_dtypes('float64').loc[:,'roomservice':'vrdeck'].sum(axis=1),
                    transported = lambda df_:df_.transported.astype('bool')
                    )
                    .drop(columns= 'cabin')
                    .set_index('passengerid')
        )
'''END FUNCIONES'''

tabla_base = pd.read_csv("D:/jafet/descargas/train.csv")

tmp_00 = tweak_df(tabla_base)

tmp_00.info()
# tmp_00.to_csv("D:/jafet/descargas/Python_codes/tmp00.csv")

'''
esta balanceado el train.
'''

tmp_00.select_dtypes('float64').hist(bins=30)
plt.show()

tmp_00.describe()
corr_matrix = tmp_00.corr(method='pearson',numeric_only=True)
corr_matrix['transported'].sort_values(ascending=False)

scatter_matrix(tmp_00.select_dtypes('float64'))
plt.show()

'''
por ser primer intento no stratificaremos 
los samples usados para train, validate, test
, tambíen la variable target esta balanceada.

Solo acepta valores númericos
tenemos que cambiar todo lo texto y tal vez todo lo de booleanos 
a números.
No acepta valores que contengan nan inf , - inf
tenemos que imputar los valores nan.
'''
'''
Haciendolo con get dummies
'''
tmp_01 =  pd.get_dummies(tmp_00,columns=['homeplanet','destination','cabin_1','cabin_2'])
tmp_01.columns

names = tmp_01.name
X = tmp_01.loc[:,~tmp_01.columns.isin(['transported','name','roomservice','foodcourt','shoppingmall','spa','vrdeck'])]
X = tmp_01.loc[:,~tmp_01.columns.isin(['transported','name'])]
Y = tmp_01.loc[:,'transported']

X_train,X_valid,Y_train,Y_valid = train_test_split(X,Y,test_size=0.2,random_state=1234)

X_train.info()

Y_train.value_counts(dropna=False)


logic_reg = LogisticRegression()
random_forest = RandomForestClassifier()
xg_boost = XGBClassifier(n_estimators = 2, max_depth = 2, learning_rate=1, objective='binary:logistic')

logic_reg.fit(X_train,Y_train)
random_forest.fit(X_train,Y_train)
xg_boost.fit(X_train,Y_train)


ex_1_tr =logic_reg.predict(X_train)
ex_2_tr = random_forest.predict(X_train)
ex_3_tr = xg_boost.predict(X_train)

confusion_matrix(Y_train,ex_1_tr)
accuracy_score(Y_train,ex_1_tr)

confusion_matrix(Y_train,ex_2_tr)
accuracy_score(Y_train,ex_2_tr)
'''
RESULTADOS DEL ENTRENAMIENTO:
>>> confusion_matrix(Y_train,ex_1_tr)
array([[2690,  747],
       [1159, 2358]], dtype=int64)
>>> accuracy_score(Y_train,ex_1_tr)
0.725913143514524
>>> confusion_matrix(Y_train,ex_2_tr)
array([[3187,  250],
       [ 124, 3393]], dtype=int64)
>>> accuracy_score(Y_train,ex_2_tr)
0.9462180040264596
'''

ex_1 =logic_reg.predict(X_valid)
ex_2 = random_forest.predict(X_valid)
ex_3 = xg_boost.predict(X_valid)

confusion_matrix(Y_valid,ex_1)
accuracy_score(Y_valid,ex_1)

confusion_matrix(Y_valid,ex_2)
accuracy_score(Y_valid,ex_2)

confusion_matrix(Y_valid,ex_3)
accuracy_score(Y_valid,ex_3)
'''
RESULTADOS VALIDACION
>>> confusion_matrix(Y_valid,ex_1)
array([[696, 182],
       [289, 572]], dtype=int64)
>>> accuracy_score(Y_valid,ex_1)
0.7291546866014951
>>> confusion_matrix(Y_valid,ex_2)
array([[650, 228],
       [286, 575]], dtype=int64)
>>> accuracy_score(Y_valid,ex_2)
0.7044278320874066

'''
'''
Probaremos cross-validation a ver que pasa

'''
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.metrics import SCORERS
sorted(sklearn.metrcs.SCORERS.keys())
SCORERS.keys()

scores = cross_val_score(random_forest, X_train, Y_train, scoring= 'f1',cv=10)

def scores_print(scores):
    print('scores',scores)
    print('mean',scores.mean())
    print('std',scores.std())

scores_print(scores)
'''
Haciendolo con spare matrix & onehotencodr
'''
names = tmp_00.name
X = tmp_00.loc[:,~tmp_00.columns.isin(['transported','name','roomservice','foodcourt','shoppingmall','spa','vrdeck'])]
X = tmp_00.loc[:,~tmp_00.columns.isin(['transported','name'])]
Y = tmp_00.loc[:,'transported']

X_train,X_valid,Y_train,Y_valid = train_test_split(X,Y,test_size=0.2,random_state=1234)

X_train.info()

# vamos a preparar nuestros valores de entrenamiento con encoders 
cat_attribs = X_train.select_dtypes('category').columns

full_pipeline = ColumnTransformer([
                                    ('cat',OneHotEncoder(),cat_attribs)
                                ])

X_train_comp = full_pipeline.fit_transform(X_train)
X_valid_comp = full_pipeline.transform(X_valid)

logic_reg = LogisticRegression()
random_forest = RandomForestClassifier()

logic_reg.fit(X_train_comp,Y_train)
random_forest.fit(X_train_comp,Y_train)


ex_1 =logic_reg.predict(X_valid_comp)
ex_2 = random_forest.predict(X_valid_comp)


confusion_matrix(Y_valid,ex_1)
accuracy_score(Y_valid,ex_1)

confusion_matrix(Y_valid,ex_2)
accuracy_score(Y_valid,ex_2)
