'''
Este es un códig para realizar las pruebas de 
el libro de time series que estamos viendo 
ya que esta en R pero no queremos R por el momento
'''

'''
Cap1
For cases 3 and 4 in Section 1.5, 
list the possible predictor variables that might be useful, 
assuming that the relevant data are available.
R: Podría sr los diferentes costos, costos por adquirencia 
costos por almacenamiento, costos por adquisición de cliente,
costos por mantenimiento, costos por venta, iva ,
descripción del producto.
For case 3 in Section 1.5, describe the five steps of forecasting in the context of this project.
R: The first step is analazying the problem and be specific on the problem

the second step is gathering info, so the case alredy says to us that they have
information, 
the third step is EDA analysis, 
The fourth is modelling
the fith is output
'''

'''
Cap2
'''
import pandas as pd
 
data = {'Year' : [2015,2016,2017,2018,2019],'Observation': [123,39,78,52,110] }

df = pd.DataFrame(data,index=data['Year'],).drop(columns='Year')
df


